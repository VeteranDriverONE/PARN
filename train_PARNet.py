import torch
import vali
import numpy as np
import surface_distance as surfdist
import warnings
import cv2
import time
from pathlib import Path
from CHAOSgenerator import CHAOS4
from torchvision import utils as vutils
from config import args
from models.PARNet import PARNet
from collections import defaultdict
from pytorch_ssim import SSIM
from thop import profile
from skimage.metrics import peak_signal_noise_ratio as psnr


def test_detail(model, args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')  
    jpg_path = Path('reg_test_img')

    test_dataset = CHAOS4(Path('E:\\datasets\\CHAOS\\Test_Sets1'), resize=(32,128,128), flag=2, label_flag=[0])

    data_per_batch = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=0)

    model.to(device)
    model.eval()
    
    p_dice = defaultdict(list)
    asd = defaultdict(list)
    hd_95 = defaultdict(list)
    so = defaultdict(list)
    sd = defaultdict(list)
    v_dice = defaultdict(list)
    j_phis = defaultdict(list)
    psnr_dict = defaultdict(list)
    ssim_dict = defaultdict(list)
    rmse_dict = defaultdict(list)

    ssim = SSIM()

    thresholds = np.linspace(0,1,101)
    
    for m_image, f_image, m_label, f_label, m_type, f_type, m_name, f_name, \
        m_origin, f_origin, m_sp, f_sp in data_per_batch:
        m_image = m_image.to(device).float()
        f_image = f_image.to(device).float()
        m_label = m_label.to(device).float()
        f_label = f_label.to(device).float()

        with torch.no_grad():
            results = model.infer(m_image, f_image, m_type, f_type)
            gen_img = results[0][0]
            warpeds = results[0][1:]
            flows = results[1]

        init_lab = m_label
        init_img = m_image
        warped_aff_lab = model.stn(init_lab, flows[0])
        warped_aff_img = model.stn(init_img, flows[0])
        # warped_aff_img = stn(gen_img, flows[0])
        # warped_lab = warped_aff_lab
        # warped_img = warped_aff_img

        for i in range(1, len(flows)):
            warped_lab = model.stn(warped_aff_lab, flows[i])
            warped_img = model.stn(warped_aff_img, flows[i])
        
        # warped_img = warpeds[-1]

        # out_img
        if True:
            jpg_path.mkdir(exist_ok=True)
            for b in range(len(m_type)):
                out_dir = Path(jpg_path) / (m_type[b] + '-' + f_type[b])
                out_dir.mkdir(exist_ok=True)
                out_dir_name = out_dir / m_name[b]
                out_dir_name.mkdir(exist_ok=True)

                for z in range(m_image.shape[2]):
                    vutils.save_image(m_image[b,0,z], str(out_dir_name / f'{z}_m.jpg'), normalize=True)
                    vutils.save_image(f_image[b,0,z], str(out_dir_name / f'{z}_f.jpg'), normalize=True)
                    vutils.save_image(m_label[b,0,z], str(out_dir_name / f'{z}_m_l.jpg'), normalize=True)
                    vutils.save_image(f_label[b,0,z], str(out_dir_name / f'{z}_f_l.jpg'), normalize=True)
                    vutils.save_image(gen_img[b,0,z], str(out_dir_name / f'{z}_t.jpg'), normalize=True)
                    # vutils.save_image(tmp_warped_aff_img[b,0,z], str(out_dir_name / f'{z}_a.jpg'), normalize=True)
                    vutils.save_image(warped_img[b,0,z], str(out_dir_name / f'{z}_w.jpg'), normalize=True)

        m_label = m_label.cpu().numpy()
        f_label = f_label.cpu().numpy()
        # warped_lab = warped_aff_lab.cpu().numpy()
        warped_lab = warped_lab.cpu().numpy()

        for b in range(len(m_type)):
            
            j_phi_perc = 0
            for i in range(len(flows)):
                j_phi = vali.NJD(flows[i][b].permute(1,2,3,0).cpu().numpy())
                j_phi_perc += j_phi / np.prod(flows[i][b].shape)
            
            ssim_val = 0
            for z in range(f_image.shape[2]):
                ssim_val += ssim.forward(warped_img[b,0,z].unsqueeze(0).unsqueeze(0), f_image[b,0,z].unsqueeze(0).unsqueeze(0))
            ssim_val = ssim_val / f_image.shape[2]
            psnr_val = psnr(f_image[b:b+1].cpu().numpy(), warped_img[b:b+1].cpu().numpy(), data_range=1)
            rmse_val = torch.nn.functional.mse_loss(warped_img[b:b+1], f_image[b:b+1]).sqrt()

            key = m_type[b] + '_' + f_type[b]
            dice_ = []
            for threshold in thresholds:
                dice_.append(vali.dice2(warped_lab[b:b+1], f_label[b:b+1], threshold))
            index = np.array(dice_).argmax()
            moving_lab = warped_lab[b:b+1]>thresholds[index]

            out_dir = Path(jpg_path) / (m_type[b] + '-' + f_type[b])
            out_dir_name = out_dir / m_name[b]
            for z in range(m_image.shape[2]):
                # vutils.save_image(warped_aff_img[b,0,z], str(out_dir_name / f'{z}_a.jpg'), normalize=True)
                tmp_dice = vali.dice_coeff(moving_lab[b,0,z:z+1], f_label[b,0,z:z+1])
                cv2.imwrite(str(out_dir_name / '{}_w_l_{:.4f}.jpg'.format(z, tmp_dice)), moving_lab[b,0,z]*255)
                cv2.imwrite(str(out_dir_name / '{}_w_l.jpg'.format(z, tmp_dice)), moving_lab[b,0,z]*255)

            moving_lab_np = moving_lab.squeeze(0).squeeze(0)
            fixed_lab_np = f_label.squeeze(0).squeeze(0).astype('bool')
            
            # spacing_mm = torch.tensor(m_origin.shape[-3:])*m_sp / torch.tensor([64,64,64])
            surface_distances = surfdist.compute_surface_distances(fixed_lab_np, moving_lab_np, spacing_mm=m_sp)
            avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances) # len=2
            hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95) # len=1
            surface_overlap = surfdist.compute_surface_overlap_at_tolerance(surface_distances, 1) # len=2
            surface_dice = surfdist.compute_surface_dice_at_tolerance(surface_distances, 1) # len=1
            volume_dice = surfdist.compute_dice_coefficient(fixed_lab_np.astype('int'), moving_lab_np.astype('int')) # len=1

            p_dice[key].append(dice_)
            asd[key].append(avg_surf_dist)
            hd_95[key].append(hd_dist_95)
            so[key].append(surface_overlap)
            sd[key].append(surface_dice)
            v_dice[key].append(volume_dice)
            j_phis[key].append(j_phi_perc)
            psnr_dict[key].append(psnr_val)
            ssim_dict[key].append(ssim_val.cpu().numpy())
            rmse_dict[key].append(rmse_val.cpu().numpy())

    all_info = ''
    dice_info = ''
    psnr_info = ''
    ssim_info = ''
    rmse_info = ''
    for k in v_dice.keys():
        print(f'\n{k}:')
        p_dice_np = np.array(p_dice[k])
        asd_np = np.array(asd[k])
        hd_95_np = np.array(hd_95[k])
        so_np = np.array(so[k])
        sd_np = np.array(sd[k])
        v_dice_np = np.array(v_dice[k])
        j_phis_np = np.array(j_phis[k])
        psnr_np = np.array(psnr_dict[k])
        ssim_np = np.array(ssim_dict[k])
        rmse_np = np.array(rmse_dict[k])

        static_p_dice = [p_dice_np.mean(0), p_dice_np.std(0)]
        static_asd = [asd_np[:,0].mean(), asd_np[:,0].std(),asd_np[:,1].mean(), asd_np[:,1].std()]
        static_hd_95 = [hd_95_np.mean(), hd_95_np.std()]
        static_so = [so_np[:,0].mean(), so_np[:,0].std(),so_np[:,1].mean(), so_np[:,1].std()]
        static_sd = [sd_np.mean(), sd_np.std()]
        static_v_dice = [v_dice_np.mean(), v_dice_np.std()]
        static_j_phis = [j_phis_np.mean(), j_phis_np.std()]
        static_psnr = psnr_np.mean()
        static_ssim = ssim_np.mean()
        static_rmse = rmse_np.mean()

        print(static_asd)
        print(static_hd_95)
        print(static_so)
        print(static_sd)
        print(static_v_dice)
        print(static_j_phis)
        print(static_psnr)
        print(static_ssim)
        print(static_rmse)

        dice_info = dice_info + k + "," + str(static_p_dice[0].squeeze(-1))[1:-1].replace('\n','') .replace(' ',',').replace(',,',',').replace(',,',',')+ '\n'
        all_info = all_info + '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(k,
                    static_asd[0],static_asd[1],static_asd[2],static_asd[3], static_hd_95[0], static_hd_95[1], static_so[0], static_so[1], static_so[2], static_so[3], 
                    static_sd[0], static_sd[1],static_v_dice[0], static_v_dice[1], static_j_phis[0], static_j_phis[1], static_psnr, static_ssim, static_rmse)
        
        psnr_info += k
        ssim_info += k
        rmse_info += k
        for i in range(len(asd[k])):
            psnr_info = psnr_info +  ',' + str(psnr_dict[k][i])
            ssim_info = ssim_info + ',' + str(ssim_dict[k][i])
            rmse_info = rmse_info + ',' + str(rmse_dict[k][i])
        psnr_info += '\n'
        ssim_info += '\n'
        rmse_info += '\n'
        
    with open('log_test_detail.csv', 'a') as f:  # 设置文件对象
        print(all_info, flush=True, file = f)
    with open('log_test_dice_detail.csv', 'a') as f:  # 设置文件对象
        print(dice_info, flush=True, file = f)
    with open('log_test_psrn_detail.csv', 'a') as f:  # 设置文件对象
        print(psnr_info, flush=True, file = f)  
    with open('log_test_ssim_detail.csv', 'a') as f:  # 设置文件对象
        print(ssim_info, flush=True, file = f)
    with open('log_test_rmse_detail.csv', 'a') as f:  # 设置文件对象
        print(rmse_info, flush=True, file = f) 


def cal_FPS(model, args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')  

    test_dataset = CHAOS4(Path('E:\\datasets\\CHAOS\\Test_Sets1'), resize=(32,128,128), flag=2, label_flag=[0])
    data_per_batch = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=0)

    model.eval()
    model = model.cuda()
    with torch.no_grad():
        for i in range(5):
            start = time.perf_counter()
            for m_image, f_image, m_label, f_label, m_type, f_type, m_name, f_name, \
                m_origin, f_origin, m_sp, f_sp in data_per_batch:
                m_image = m_image.float().to(device)
                f_image = f_image.float().to(device)
                m_label = m_label.float().to(device)
                f_label = f_label.float().to(device)     
                model.infer(m_image, f_image, m_type, f_type)
            end = time.perf_counter()
            print(f'Total inference time:{end-start}, Average inference time:{(end-start)/len(test_dataset)}')


if __name__ == '__main__':
    # Kindey tumor registration hyperparameters
    # args.w_dis = 1
    # args.w_gan = 1
    # args.w_style = 1
    # args.w_trans_sim = 0.1
    # args.w_consis = 1
    # args.w_proto = 1
    # args.contra = 1
    # args.w_arg = 1
    # args.w_smooth = 1

    # CHAOS hyperparameters
    args.w_dis = 1
    args.w_gan = 1
    args.w_style = 1
    args.w_trans_sim = 1
    args.w_consis = 5
    args.w_proto = 1e-2
    args.w_contra = 1
    args.w_reg = 1
    args.w_smooth = 1
    
    if args.root_path is None:
        args.root_path = Path.cwd()

    parnet = PARNet(args)
    if args.load_model is not None:
        parnet.load_model(args.load_model, is_best=False)
    # parnet.train_dataset()

    if args.test_load_model is not None:
        parnet.load_model(args.test_load_model, is_best=True)
    test_detail(parnet, args)

    cal_FPS(parnet, args)