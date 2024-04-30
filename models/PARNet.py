import torch
import numpy as np
import time
import vali
import models.losses as losses 

from pathlib import Path
from collections import defaultdict
from models.UNet5 import Encoder, Decoder, SpatialTransformer, Dis
from models.utils import FlowShow
from models.base_networks import VTNAffineStem, VTN, VxmDense
from CHAOSgenerator import CHAOS4
from .modelio import LoadableModel, store_config_args
from config import args

from torch.utils.tensorboard import SummaryWriter


class PARNet(LoadableModel):
    @store_config_args
    def __init__(self,args, n_cascades=1):
        super(PARNet,self).__init__()
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu' )
        self.stems = []
        
        self.dis = Dis().to(self.device)
        self.encoder = Encoder().to(self.device)
        self.gen = Decoder().to(self.device)
        id_map = torch.FloatTensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(self.device)
        self.affine = VTNAffineStem(dim=len(args.img_shape), im_size=args.img_shape, id_map=id_map).to(self.device)

        trainable_params = list(self.affine.parameters())
        for i in range(n_cascades):
            cas = VTN(dim=len(args.img_shape), flow_multiplier=1.0 / n_cascades).to(self.device)
            # cas = VxmDense().to(self.device)
            self.stems.append(cas)
            trainable_params += list(cas.parameters())
        self.stn = SpatialTransformer(args.img_shape).to(self.device)

        self.optimizerD = torch.optim.Adam(self.dis.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizerE = torch.optim.Adam(self.encoder.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizerG = torch.optim.Adam(self.gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizerR = torch.optim.Adam(trainable_params, lr=args.lr, betas=(0.5, 0.999))

        self.recoder = defaultdict(list)
        self.writer = SummaryWriter(args.train_tb)
        self.flowshow = FlowShow(args.img_shape, 'grid_pic.jpg', self.device)
        self.now_epoch = 0
        # self.mse = torch.nn.MSELoss()
        self.mse = torch.nn.L1Loss()
        self.ncc = losses.NCC().loss
        self.smooth = losses.regularize_loss_3d
        
    def load_model(self, load_path, is_best=True):
        print('load saved model')
        ckp = torch.load(load_path, map_location=torch.device(self.device))        
        self.dis.load_state_dict(ckp['dis'])
        self.encoder.load_state_dict(ckp['encoder'])
        self.gen.load_state_dict(ckp['gen'])
        self.affine.load_state_dict(ckp['affine'])
        for i in range(len(self.stems)):
            self.stems[i].load_state_dict(ckp[f'cascade_{i}'])
        if not is_best:
            self.now_epoch = ckp['now_epoch']
            self.optimizerD.load_state_dict(ckp['optimizerD'])
            self.optimizerE.load_state_dict(ckp['optimizerE'])
            self.optimizerG.load_state_dict(ckp['optimizerG'])
            self.optimizerR.load_state_dict(ckp['optimizerR'])

        # for key in cpk.keys():
        #     obj = getattr(self, key)
        #     if isinstance(obj, torch.nn.Module) or isinstance(obj, torch.optim.Optimizer):
        #         obj.load_state_dict(cpk[key])
        #     else:
        #         setattr(self, key, cpk[key])

    def save_model(self, epoch=0, is_best=False):
        ckp = {}
        ckp['dis'] = self.dis.state_dict()
        ckp['encoder'] = self.encoder.state_dict()
        ckp['gen'] = self.gen.state_dict()
        ckp['affine'] = self.affine.state_dict()
        for i in range(len(self.stems)):
            ckp[f'cascade_{i}'] = self.stems[i].state_dict()
        
        if is_best:
            torch.save(ckp, Path(args.root_path) / args.model_dir / 'parnet_best.pt')
            return 
        
        ckp['optimizerD'] = self.optimizerD.state_dict()
        ckp['optimizerE'] = self.optimizerE.state_dict()
        ckp['optimizerG'] = self.optimizerG.state_dict()
        ckp['optimizerR'] = self.optimizerR.state_dict()
        ckp['now_epoch'] = epoch
        torch.save(ckp, Path(args.root_path) / args.model_dir / f'parnet_{epoch}.pt')

    def write_tb(self, epoch):
        for k, v in self.recoder.items():
            loss_np = np.array(v)
            self.writer.add_scalar(f'train/{k}', loss_np.mean(), epoch)

    def print_loss(self, epoch_info, is_write=False):
        count = 1
        loss_info = ''
        for k, v in self.recoder.items():
            loss_info = loss_info + '{}:{:.4e},'.format(k, np.array(v).mean())
            if count % 4 == 0:
                loss_info = loss_info + '\n'
            count += 1

        print(loss_info)
        if is_write:
            with open('log_train_loss.txt', 'a') as f:  # 设置文件对象
                print(epoch_info, flush=True, file = f)
                print(loss_info, flush=True, file = f)
            
    def clear_recoder(self):
        self.record = defaultdict(list)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def back_dis(self, fixeds):
        dis_loss = self.dis.dis_loss(self.gen_img.detach(), fixeds)*self.args.w_dis
        dis_loss.backward()
        self.recoder['dis_loss'].append(dis_loss.item()*self.args.w_dis)

    def back_gen(self):
        encode_x = self.encoder.forward1(self.gen_img)
        style_loss = 0
        for i in range(2, len(encode_x)):
            style_loss = style_loss +  self.mse(encode_x[i], self.skip_f[i])*self.args.w_style
        style_loss.backward(retain_graph=True)
        self.recoder['style_loss'].append(style_loss.item()*self.args.w_style)

        self.total_loss = style_loss.item()
        
    def back_trans(self, moving, fixed):
        gan_loss = self.dis.gen_loss(self.gen_img)
        sim_loss1 = self.mse(self.gen_img, fixed)
        consis_loss = self.ncc(moving, self.gen_img)
        trans_loss = gan_loss*self.args.w_gan + sim_loss1*self.args.w_trans_sim + consis_loss*self.args.w_consis + self.encoder.prototype_loss*self.args.w_proto + self.encoder.contra*self.args.w_contra
        trans_loss.backward(retain_graph=True)

        self.recoder['gan_gen'].append(gan_loss.item()*self.args.w_gan)
        self.recoder['sim_loss1'].append(sim_loss1.item()*self.args.w_trans_sim)
        self.recoder['consis_loss'].append(consis_loss.item()*self.args.w_consis)
        self.recoder['prototype_loss'].append(self.encoder.prototype_loss.item()*self.args.w_proto)
        self.recoder['contra_loss'].append(self.encoder.contra.item()*self.args.w_contra)

        self.total_loss += trans_loss.item()

    def back_reg(self, fixed):
        # affine registration
        sim_loss2 = self.mse(self.warpeds[0], fixed)
        smooth_loss2 = self.smooth(self.flows[0])
        
        # deformable registration
        sim_loss3 = 0
        smooth_loss3 = 0
        for i in range(len(self.stems)):
            # sim_loss3 = sim_loss3 + self.ncc(fixed, self.warpeds[2+i])
            sim_loss3 = sim_loss3 + self.mse(self.warpeds[1+i], fixed)
            smooth_loss3 = smooth_loss3 + self.smooth(self.flows[1+i])

        reg_loss = (sim_loss2 + sim_loss3)*self.args.w_reg + (smooth_loss2 + smooth_loss3)*self.args.w_smooth
        reg_loss.backward()
        
        self.total_loss += reg_loss.item()

        self.recoder['sim_loss2'].append(sim_loss2.item()*self.args.w_reg)
        self.recoder['sim_loss3'].append(sim_loss3.item()*self.args.w_reg)
        self.recoder['smooth_loss2'].append(smooth_loss2.item()*self.args.w_smooth)
        self.recoder['smooth_loss3'].append(smooth_loss3.item()*self.args.w_smooth)
        self.recoder['reg_loss'].append(reg_loss.item())

        self.recoder['total_loss'].append(self.total_loss)

    def optimize(self, m_image, f_image, m_type, f_type):
        self.global_step = self.global_step + 1
        self.forward(m_image, f_image, m_type, f_type)

        # back dis
        self.optimizerD.zero_grad()
        self.set_requires_grad([self.encoder, self.gen], requires_grad=False)
        self.back_dis(f_image)

        # back gen
        self.optimizerG.zero_grad()
        self.set_requires_grad([self.gen], requires_grad=True)
        self.back_gen()

        # back trans
        self.optimizerE.zero_grad()
        self.set_requires_grad([self.encoder], requires_grad=True)
        self.back_trans(m_image, f_image)
        # torch.nn.utils.clip_grad_norm_(list(self.gen.parameters()), max_norm=20, norm_type=2)
        
        # back reg
        self.optimizerR.zero_grad()
        self.back_reg(f_image)
        
        # torch.nn.utils.clip_grad_norm_(list(self.affine.parameters()), max_norm=20, norm_type=2)
        # param = []
        # for cas in self.stems:
        #     param = param + list(cas.parameters())
        # torch.nn.utils.clip_grad_norm_(param, max_norm=20, norm_type=2)

        self.optimizerD.step()
        self.optimizerE.step()
        self.optimizerG.step()
        self.optimizerR.step()

        if self.global_step % self.args.tb_save_freq == 0:
            self.tb_save_step += 1
            ran_id = int((torch.rand(1)*0.8+0.1) * m_image.size(2))
            self.writer.add_image('train/reg_img', (m_image[0,0,ran_id:ran_id+1,:,:]), self.tb_save_step)
            self.writer.add_image('train/reg_fixed', (f_image[0,0,ran_id:ran_id+1,:,:]), self.tb_save_step)
            self.writer.add_image(f'train/trans_img', (self.gen_img[0,0,ran_id:ran_id+1,:,:]), self.tb_save_step) 
            for i in range(len(self.warpeds)):
                self.writer.add_image(f'train/warped_{i}', (self.warpeds[i][0,0,ran_id:ran_id+1,:,:]), self.tb_save_step) 

    def forward(self,moving, fixed, m_type, f_type):
        self.skip_m, self.skip_f = self.encoder(moving, fixed, m_type, f_type)
        self.gen_img = self.gen(self.skip_m, self.skip_f, m_type, f_type, self.encoder.prototypes)

        self.flows = []
        self.warpeds = []

        flow, self.id_loss = self.affine(self.gen_img, fixed)
        self.flows.append(flow)
        self.warpeds.append(self.stn(self.gen_img, flow))

        for cas in self.stems:
            self.flows.append(cas(self.warpeds[-1], fixed))
            self.warpeds.append(self.stn(self.warpeds[-1], self.flows[-1]))
        
        # return self.warpeds, self.flows
    
    def train_dataset(self):
        
        train_dataset = CHAOS4(Path('E:\\datasets\\CHAOS\\Train_Sets1'), resize=(32,128,128), flag=0)
        val_dataset = CHAOS4(Path('E:\\datasets\\CHAOS\\Test_Sets1'), resize=(32,128,128), flag=1)

        self.train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=0)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

        self.global_step = (self.now_epoch+1) * len(self.train_loader)
        self.tb_save_step = self.global_step // self.args.tb_save_freq
        best_dice = 0

        for epoch in range(self.now_epoch+1,args.epoch+1):
            epoch_info = "Epoch:{}/{} - ".format(epoch, args.epoch) + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            print(epoch_info)
            for movings, fixeds, m_type, f_type in self.train_loader:
                movings, fixeds = movings.float().to(self.device), fixeds.float().to(self.device)
                self.optimize(movings, fixeds, m_type, f_type)

            # 打印信息
            self.write_tb(epoch) 
            self.print_loss(epoch_info, is_write=True)
            self.clear_recoder()

            if epoch % self.args.save_per_epoch == 0:
                mean_dice = self.val_dataset(epoch)
                if mean_dice > best_dice:
                    best_dice = mean_dice
                    self.save_model(is_best=True)

            if epoch % self.args.save_per_epoch == 0:
                self.save_model(epoch=epoch)

    def val_dataset(self, epoch):
        dice_dict = defaultdict(list)
        thresholds = np.linspace(0,1,101)
        step = 0
        gap = len(self.val_dataloader)

        jpg_path = self.args.root_path / "reg_test_img" if isinstance(self.args.root_path, Path) else Path(self.args.root_path) / "reg_test_img"

        with torch.no_grad():
            dices = []
            for movings, fixeds, moving_lab, fixed_lab, m_type, f_type in self.val_dataloader:
                moving_img = movings.to(self.device).float()
                fixed_img = fixeds.to(self.device).float()
                moving_lab = moving_lab.to(self.device).float()
                fixed_lab = fixed_lab.to(self.device).float()

                self.forward(moving_img, fixed_img, m_type, f_type)

                warped_lab = moving_lab
                for i in range(1, len(self.flows)):
                    warped_lab = self.stn(warped_lab, self.flows[i])


                ran_id = torch.argmax(torch.sum(fixed_lab,(3,4)),dim=-1)[0].squeeze()
                # ran_id = int((torch.rand(1)*0.8+0.1) * input_img.size(2))
                self.writer.add_image('test_label/moving_lab', moving_lab[0,0,ran_id:ran_id+1,:,:], epoch*gap+step)
                self.writer.add_image('test_label/fixed_lab', fixed_lab[0,0,ran_id:ran_id+1,:,:], epoch*gap+step)          
                self.writer.add_image('test_label/warped_lab', warped_lab[0,0,ran_id:ran_id+1,:,:], epoch*gap+step)
                warped_fixed = (warped_lab>0)*0.5 + (fixed_lab)*0.5
                self.writer.add_image('test_label/overlap', warped_fixed[0,0,ran_id:ran_id+1,:,:], epoch*gap+step)
                warped_fixed = (moving_lab>0)*0.5 + (fixed_lab)*0.5
                self.writer.add_image('test_label/overlap_real', warped_fixed[0,0,ran_id:ran_id+1,:,:], epoch*gap+step)

                self.writer.add_image('test_image/moving_img', moving_img[0,0,ran_id:ran_id+1,:,:], epoch*gap+step)
                self.writer.add_image('test_image/fixed_img', fixed_img[0,0,ran_id:ran_id+1,:,:], epoch*gap+step)
                self.writer.add_image(f'test_image/gen_img', self.gen_img[0,0,ran_id:ran_id+1,:,:], epoch*gap+step)
                for i in range(len(self.warpeds)):
                    self.writer.add_image(f'test_image/wapred_{i}', self.warpeds[i][0,0,ran_id:ran_id+1,:,:], epoch*gap+step)
                    flow_vis = self.flowshow.show(self.stn, self.flows[i])
                    self.writer.add_image(f'test_image/flow_{i}', flow_vis[0,0,ran_id:ran_id+1,:,:]/255, epoch*gap+step)
                
                step = step + 1

                for i in range(len(m_type)):
                    key = m_type[i] + '_' + f_type[i]
                    dice_ = []
                    for threshold in thresholds:
                        dice_.append(vali.dice2_(warped_lab[i:i+1], fixed_lab[i:i+1], threshold))
                    dice_dict[key].append(dice_)

            dice_detail = f'\nepoch:{epoch}\n'
            dice_info = f'\nepoch:{epoch}\n'
            key_max = 0
            for k in dice_dict.keys():
                dice_array =  np.array(dice_dict[k])
                index = dice_array.mean(0).argmax()
                max_v = dice_array.mean(0)[index]
                std_v = dice_array.std(0)[index]
                dice_detail = dice_detail + str(dice_array.mean(0))[1:-1].replace(' ',',').replace('\n',' ') + '\n'
                dice_info = dice_info + f'{k}:{max_v},{std_v},{index}\n' 
                key_max += max_v

            with open(str(Path(self.args.root_path) / 'expriment_test.txt'), 'a') as f:  # 设置文件对象
                print(dice_detail, flush=True, file = f)
            
            print(dice_info)
            with open(str(Path(self.args.root_path) / 'log_test.txt'), 'a') as f:  # 设置文件对象
                print(dice_info, flush=True, file = f)
            
        return key_max / len(dice_dict.keys())

    def infer(self, m_image, f_image, m_type, f_type):
        with torch.no_grad():
            self.forward(m_image, f_image, m_type, f_type)
        return [self.gen_img]+self.warpeds, self.flows
