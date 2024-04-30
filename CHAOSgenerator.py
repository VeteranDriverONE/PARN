import torch
import numpy as np
import SimpleITK as sitk
import warnings
import cv2

from collections import defaultdict
from pathlib import Path

def adjustWW(image, width=None, level=None):
    # 腹部窗宽350，窗位40

    if width is None or level is None:
        max_v = image.max()
        min_v = image.min()
        voxel_num = np.prod(image.shape)
        width = max_v
        for i in range(int(max_v), int(min_v), -1):
            if (image > i).sum() / voxel_num > 0.001:
                width = i
                break

        level = width // 2

    v_min = level - (width / 2)
    v_max = level + (width / 2)

    img = image.copy()
    img[image < v_min] = v_min
    img[image > v_max] = v_max

    img = (img - v_min) / (v_max - v_min)
    # img = (img-img.mean()) / img.std()

    return img


class CHAOS4(torch.utils.data.Dataset):
    def __init__(self, root_path, fixed_seqs:list=['T1DUAL','T2SPIR'], moving_seqs:list=['T1DUAL','T2SPIR'], resize=None, flag:int=0, label_flag:list=[0]):
        # flag(train, validate or test):0:train，1:validate，2:test
        # label_flag (registration of organ):0:Liver, 1:Right kindey, 2:Left Kindey, 3:Spleen
        self.path = root_path
        label_flag = label_flag
        self.flag = flag
        label_arrange = [[55,70],[110,135],[175,200],[240,255]]  # branch:['Liver','Right Kindey','Left Kindey', 'Spleen']
        reader = sitk.ImageSeriesReader() 

        self.fixed_imgs = []
        self.moving_imgs = []
        self.fixed_labs = []
        self.moving_labs = []
        self.moving_origin = []
        self.fixed_origin = []
        self.p_types_movings = [] # CMP,NP等字符串
        self.p_types_fixeds = []
        self.movings_name = []
        self.fixeds_name =[]
        
        self.name = []
        self.type = []
        self.origin = []
        self.spacing = []
        
        seqs = list(set(fixed_seqs).union(set(moving_seqs)))
        img_seqs_dict = defaultdict(list)
        lab_seqs_dict = defaultdict(list)
        name_dict = defaultdict(list)

        # read file
        path_list = []
        for key in seqs:
            path_list += root_path.rglob(key)
        for dir in path_list:
            p_name = dir.parent.name
            if dir.name == 'T2SPIR':
                img_seqs_dict[dir.name].append(dir/"DICOM_anon")
                name_dict[dir.name].append(p_name)
            elif dir.name == 'T1DUAL':
                img_seqs_dict[dir.name].append(dir/'DICOM_anon'/'OutPhase')
                name_dict[dir.name].append(p_name)
            
            if flag > 0:
                lab_seqs_dict[dir.name].append(dir/"Ground")
        
        # read and resize
        img_seqs_dict_itk = defaultdict(list)
        lab_seqs_dict_itk = defaultdict(list)
        if flag == 0:
            for key in img_seqs_dict.keys():
                for ind in range(len(img_seqs_dict[key])):

                    img_names = reader.GetGDCMSeriesFileNames(str(img_seqs_dict[key][ind]))
                    reader.SetFileNames(img_names)
                    image_itk = reader.Execute()
                    
                    if resize is not None:
                        image_itk = self.__resize__(resize, image_itk)
                                        
                    img_seqs_dict_itk[key].append(image_itk)
        else:
            for key in img_seqs_dict.keys():
                for ind in range(len(img_seqs_dict[key])):

                    img_names = reader.GetGDCMSeriesFileNames(str(img_seqs_dict[key][ind]))
                    reader.SetFileNames(img_names)
                    image_itk = reader.Execute()

                    label_ls = []
                    pic_ls = list(lab_seqs_dict[key][ind].glob("*.png"))
                    pic_ls.sort()
                    for pic in pic_ls:
                        label_ls.append(cv2.imread(str(pic),cv2.IMREAD_GRAYSCALE))
                    label_np = np.stack(label_ls)
                    
                    new_label_np = np.zeros_like(label_np)
                    for ind in label_flag:
                        new_label_np = new_label_np + (label_np>label_arrange[ind][0])*(label_np<label_arrange[ind][1])
                        # (label_np == 63) + (label_np == 126) * (label_np == 189) * (label_np == 252)
                    assert len(np.unique(new_label_np))<=2, '类别不为0和1'

                    label_itk = sitk.GetImageFromArray(new_label_np)
                    label_itk.SetDirection(image_itk.GetDirection())
                    label_itk.SetOrigin(image_itk.GetOrigin())
                    label_itk.SetSpacing(image_itk.GetSpacing())

                    if resize is not None:
                        image_itk, label_itk = self.__resize__(resize, image_itk, label_itk)
                    
                    img_seqs_dict_itk[key].append(image_itk)
                    lab_seqs_dict_itk[key].append(label_itk)

        # pair
        for m_type in img_seqs_dict.keys():
            for f_type in img_seqs_dict.keys():
                if m_type == f_type:
                    continue
                for i in range(len(img_seqs_dict_itk[m_type])):

                    m_image_itk = img_seqs_dict_itk[m_type][i]
                    f_image_itk = img_seqs_dict_itk[f_type][i]

                    moving_img = sitk.GetArrayFromImage(m_image_itk)
                    fixed_img = sitk.GetArrayFromImage(f_image_itk)

                    if m_type == 'T1DUAL' or m_type == 'T2SPIR':
                        moving_img = adjustWW(moving_img, width=1000, level=400)
                    elif m_type == 'CT':
                        moving_img = adjustWW(moving_img, width=350, level=40)
                    else:
                        assert False, '错误的类别'
                    
                    if f_type == 'T1DUAL' or f_type == 'T2SPIR':
                        fixed_img = adjustWW(fixed_img, width=1000, level=400)
                    elif f_type == 'CT':
                        fixed_img = adjustWW(fixed_img, width=350, level=40)
                    else:
                        assert False, '错误的类别'

                    self.moving_imgs.append(moving_img)
                    self.fixed_imgs.append(fixed_img)
                    
                    m_o = m_image_itk.GetOrigin()
                    f_o = f_image_itk.GetOrigin()
                    m_sp = m_image_itk.GetSpacing()
                    f_sp = f_image_itk.GetSpacing()
                    self.origin.append([(m_o[2], m_o[0], m_o[1]), (f_o[2], f_o[0], f_o[1])])
                    self.spacing.append([(m_sp[2],m_sp[0],m_sp[1]), (f_sp[2],f_sp[0],f_sp[1])])
                    self.type.append([m_type, f_type])
                    self.name.append([name_dict[m_type][i],name_dict[f_type][i]])

                    if flag > 0:
                        m_label_itk = lab_seqs_dict_itk[m_type][i]
                        f_label_itk = lab_seqs_dict_itk[f_type][i]

                        moving_lab = sitk.GetArrayFromImage(m_label_itk).astype('uint8')
                        fixed_lab = sitk.GetArrayFromImage(f_label_itk).astype('uint8')

                        self.moving_labs.append(moving_lab)
                        self.fixed_labs.append(fixed_lab)
        
        # self.__write_img__(img_seqs_dict_itk, lab_seqs_dict_itk, name_dict, Path('E:\\datasets\\CHAOS\\output'))
    
    def __len__(self):
        return len(self.moving_imgs)

    def __resize__(self, new_size, img_itk, lab_itk=None):
        img_origin_size = np.array(img_itk.GetSize()) # w,h,z
        img_origin_spacing = np.array(img_itk.GetSpacing()) # z,h,w -> w,h,z

        new_size = np.array((new_size[2],new_size[1],new_size[0]))
        new_spacing = (img_origin_size * img_origin_spacing) / new_size

        # 图像缩放            
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img_itk)
        resampler.SetSize(new_size.tolist())
        resampler.SetOutputSpacing(new_spacing.tolist())
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        img_itk_resample = resampler.Execute(img_itk)
        
        # img = sitk.GetArrayFromImage(img_itk_resample) # w,h,z
        # new_img = sitk.GetImageFromArray(img)
        # # new_img = sitk.Cast(sitk.RescaleIntensity(new_img), sitk.sitkUInt8)
        # new_img.SetDirection(img_itk_resample.GetDirection())
        # new_img.SetOrigin(img_itk_resample.GetOrigin())
        # new_img.SetSpacing(img_itk_resample.GetSpacing())
        if lab_itk is None:
            return img_itk_resample
        
        lab_origin_size = np.array(lab_itk.GetSize()) # w,h,z
        lab_origin_spacing = np.array(lab_itk.GetSpacing()) # z,h,w -> w,h,z
        
        assert (img_origin_size == lab_origin_size).all() and (img_origin_spacing == lab_origin_spacing).all(), '图像和标签的origing和spacing不一致'

        # 标签缩放
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(lab_itk)
        resampler.SetSize(new_size.tolist())
        resampler.SetOutputSpacing(new_spacing.tolist())
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        lab_itk_resample = resampler.Execute(lab_itk)

        # lab = sitk.GetArrayFromImage(lab_itk_resample) # w,h,z
        # new_lab = sitk.GetImageFromArray(lab)
        # new_lab.SetDirection(lab_itk_resample.GetDirection())
        # new_lab.SetOrigin(lab_itk_resample.GetOrigin())
        # new_lab.SetSpacing(lab_itk_resample.GetSpacing())

        return img_itk_resample, lab_itk_resample

    def __write_img__(self, img_seqs_dict, lab_seqs_dict, name, out_path):
        for key in img_seqs_dict.keys():
            for i in range(len(img_seqs_dict[key])):
                file_name = f'{name[key][i]}_{key}_image.nii.gz'
                sitk.WriteImage(img_seqs_dict[key][i], str(out_path/file_name))
                
                if self.flag > 0:
                    file_name = f'{name[key][i]}_{key}_label.nii.gz'
                    sitk.WriteImage(lab_seqs_dict[key][i], str(out_path/file_name))

    def __getitem__(self, index):
        if self.flag == 0:
            moving_img = torch.tensor(self.moving_imgs[index])
            fixed_img = torch.tensor(self.fixed_imgs[index])
            m_type, f_type = self.type[index]
            return moving_img.unsqueeze(0), fixed_img.unsqueeze(0), m_type, f_type
        elif self.flag == 1:
            m_img = torch.tensor(self.moving_imgs[index]) 
            f_img = torch.tensor(self.fixed_imgs[index]) 
            m_lab = torch.tensor(self.moving_labs[index])
            f_lab = torch.tensor(self.fixed_labs[index])
            m_type, f_type = self.type[index]
            return m_img.unsqueeze(0), f_img.unsqueeze(0), m_lab.unsqueeze(0), f_lab.unsqueeze(0), m_type, f_type
        elif self.flag == 2 :
            m_img = torch.tensor(self.moving_imgs[index]) 
            f_img = torch.tensor(self.fixed_imgs[index]) 
            m_lab = torch.tensor(self.moving_labs[index])
            f_lab = torch.tensor(self.fixed_labs[index])
            m_name, f_name = self.name[index]
            m_type, f_type = self.type[index]
            m_origin, f_origin = self.origin[index]
            m_spacing, f_spacing = self.spacing[index]
            return m_img.unsqueeze(0), f_img.unsqueeze(0), m_lab.unsqueeze(0), f_lab.unsqueeze(0), \
                m_type, f_type, m_name, f_name, m_origin, f_origin, m_spacing, f_spacing
        else:
            warnings.warn(f'不存在的flag选项{self.flag}')
