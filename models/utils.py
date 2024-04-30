# internal python imports
import os
import csv
import functools
import torch

# import itk
# import torch.nn.functional as F
# import pygem
# print(pygem.__version__)
# from pygem import FFD

# third party imports
import numpy as np
import scipy
from skimage import measure
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
# from UNet5 import SpatialTransformer


# local/our imports
# import pystrum.pynd.ndutils as nd

class SpatialTransformer(torch.nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        # src B,C,Z,Y,Z
        # flow 3,z,y,x
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        # return F.grid_sample(src, new_locs, mode=self.mode)
        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=False)

def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features


def get_backend():
    """
    Returns the currently used backend. Default is tensorflow unless the
    VXM_BACKEND environment variable is set to 'pytorch'.
    """
    return 'pytorch' if os.environ.get('VXM_BACKEND') == 'pytorch' else 'tensorflow'


def read_file_list(filename, prefix=None, suffix=None):
    '''
    Reads a list of files from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    with open(filename, 'r') as file:
        content = file.readlines()
    filelist = [x.strip() for x in content if x.strip()]
    if prefix is not None:
        filelist = [prefix + f for f in filelist]
    if suffix is not None:
        filelist = [f + suffix for f in filelist]
    return filelist


def read_pair_list(filename, delim=None, prefix=None, suffix=None):
    '''
    Reads a list of registration file pairs from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        delim: File pair delimiter. Default is a whitespace seperator (None).
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    pairlist = [f.split(delim) for f in read_file_list(filename)]
    if prefix is not None:
        pairlist = [[prefix + f for f in pair] for pair in pairlist]
    if suffix is not None:
        pairlist = [[f + suffix for f in pair] for pair in pairlist]
    return pairlist


def load_volfile(
    filename,
    np_var='vol',
    add_batch_axis=False,
    add_feat_axis=False,
    pad_shape=None,
    resize_factor=1,
    ret_affine=False
):
    """
    Loads a file in nii, nii.gz, mgz, npz, or npy format. If input file is not a string,
    returns it directly (allows files preloaded in memory to be passed to a generator)

    Parameters:
        filename: Filename to load, or preloaded volume to be returned.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol'.
        add_batch_axis: Adds an axis to the beginning of the array. Default is False.
        add_feat_axis: Adds an axis to the end of the array. Default is False.
        pad_shape: Zero-pad the array to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    """
    if isinstance(filename, str) and not os.path.isfile(filename):
        raise ValueError("'%s' is not a file." % filename)

    if not os.path.isfile(filename):
        if ret_affine:
            (vol, affine) = filename
        else:
            vol = filename
    elif filename.endswith(('.nii', '.nii.gz', '.mgz')):
        import nibabel as nib
        img = nib.load(filename)
        vol = img.get_data().squeeze()
        affine = img.affine
    elif filename.endswith('.npy'):
        vol = np.load(filename)
        affine = None
    elif filename.endswith('.npz'):
        npz = np.load(filename)
        vol = next(iter(npz.values())) if len(npz.keys()) == 1 else npz[np_var]
        affine = None
    else:
        raise ValueError('unknown filetype for %s' % filename)

    if pad_shape:
        vol, _ = pad(vol, pad_shape)

    if add_feat_axis:
        vol = vol[..., np.newaxis]

    if resize_factor != 1:
        vol = resize(vol, resize_factor)

    if add_batch_axis:
        vol = vol[np.newaxis, ...]

    return (vol, affine) if ret_affine else vol

class FlowShow():
    def __init__(self, shape, grid_path, device):
        z, h, w = shape
        grid_pic = cv2.imread(grid_path,2)[np.newaxis,np.newaxis,...] # 1, 1, h, w
        self.grid_pic = torch.from_numpy(grid_pic).repeat(1,z,1,1).float().to(device)

    def show(self, stn, flow):
        b = flow.size(0)        
        return stn(self.grid_pic.unsqueeze(0).repeat(b,1,1,1,1), flow)

def create_grid(out_path, size=(128,128)):
    num1, num2 = (size[0]+10) // 10, (size[1]+10) // 10  # 改变除数（10），即可改变网格的密度
    x, y = np.meshgrid(np.linspace(-2, 2, num1), np.linspace(-2, 2, num2))

    plt.figure(figsize=((size[0]) / 100.0, (size[1]) / 100.0))  # 指定图像大小
    plt.plot(x, y, color="black")
    plt.plot(x.transpose(), y.transpose(), color="black")
    plt.axis('off')  # 不显示坐标轴
    # 去除白色边框
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(out_path)  # 保存图像
    # plt.show()

def save_volfile(array, filename, affine=None):
    """
    Saves an array to nii, nii.gz, or npz format.

    Parameters:
        array: The array to save.
        filename: Filename to save to.
        affine: Affine vox-to-ras matrix. Saves LIA matrix if None (default).
    """
    if filename.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        if affine is None and array.ndim >= 3:
            # use LIA transform as default affine
            affine = np.array([[-1, 0, 0, 0],  # nopep8
                               [0, 0, 1, 0],  # nopep8
                               [0, -1, 0, 0],  # nopep8
                               [0, 0, 0, 1]], dtype=float)  # nopep8
            pcrs = np.append(np.array(array.shape[:3]) / 2, 1)
            affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
        nib.save(nib.Nifti1Image(array, affine), filename)
    elif filename.endswith('.npz'):
        np.savez_compressed(filename, vol=array)
    else:
        raise ValueError('unknown filetype for %s' % filename)


def load_labels(arg):
    """
    Load label maps and return a list of unique labels as well as all maps.

    Parameters:
        arg: Path to folder containing label maps, string for globbing, or a list of these.

    Returns:
        np.array: List of unique labels.
        list: List of label maps, each as a np.array.
    """
    if not isinstance(arg, (tuple, list)):
        arg = [arg]

    # List files.
    import glob
    ext = ('.nii.gz', '.nii', '.mgz', '.npy', '.npz')
    files = [os.path.join(f, '*') if os.path.isdir(f) else f for f in arg]
    files = sum((glob.glob(f) for f in files), [])
    files = [f for f in files if f.endswith(ext)]

    # Load labels.
    if len(files) == 0:
        raise ValueError(f'no labels found for argument "{files}"')
    label_maps = []
    shape = None
    for f in files:
        x = np.squeeze(load_volfile(f))
        if shape is None:
            shape = np.shape(x)
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError(f'file "{f}" has non-integral data type')
        if not np.all(x.shape == shape):
            raise ValueError(f'shape {x.shape} of file "{f}" is not {shape}')
        label_maps.append(x)

    return np.unique(label_maps), label_maps


def load_pheno_csv(filename, training_files=None):
    """
    Loads an attribute csv file into a dictionary. Each line in the csv should represent
    attributes for a single training file and should be formatted as:

    filename,attr1,attr2,attr2...

    Where filename is the file basename and each attr is a floating point number. If
    a list of training_files is specified, the dictionary file keys will be updated
    to match the paths specified in the list. Any training files not found in the
    loaded dictionary are pruned.
    """

    # load csv into dictionary
    pheno = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        for row in csv_reader:
            pheno[row[0]] = np.array([float(f) for f in row[1:]])

    # make list of valid training files
    if training_files is None:
        training_files = list(training_files.keys())
    else:
        training_files = [f for f in training_files if os.path.basename(f) in pheno.keys()]
        # make sure pheno dictionary includes the correct path to training data
        for f in training_files:
            pheno[f] = pheno[os.path.basename(f)]

    return pheno, training_files


def pad(array, shape):
    """
    Zero-pads an array to a given shape. Returns the padded array and crop slices.
    """
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
    padded[slices] = array

    return padded, slices


def resize(array, factor, batch_axis=False):
    """
    Resizes an array by a given factor. This expects the input array to include a feature dimension.
    Use batch_axis=True to avoid resizing the first (batch) dimension.
    """
    if factor == 1:
        return array
    else:
        if not batch_axis:
            dim_factors = [factor for _ in array.shape[:-1]] + [1]
        else:
            dim_factors = [1] + [factor for _ in array.shape[1:-1]] + [1]
        return scipy.ndimage.interpolation.zoom(array, dim_factors, order=0)


def dice(array1, array2, labels=None, include_zero=False):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.

    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute dice on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0)) 

    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem


def affine_shift_to_matrix(trf, resize=None, unshift_shape=None):
    """
    Converts an affine shift to a matrix (over the identity).
    To convert back from center-shifted transform, provide image shape
    to unshift_shape.

    TODO: make ND compatible - currently just 3D
    """
    matrix = np.concatenate([trf.reshape((3, 4)), np.zeros((1, 4))], 0) + np.eye(4)
    if resize is not None:
        matrix[:3, -1] *= resize
    if unshift_shape is not None:
        T = np.zeros((4, 4))
        T[:3, 3] = (np.array(unshift_shape) - 1) / 2
        matrix = (np.eye(4) + T) @ matrix @ (np.eye(4) - T)
    return matrix


def extract_largest_vol(bw, connectivity=1):
    """
    Extracts the binary (boolean) image with just the largest component.
    TODO: This might be less than efficiently implemented.
    """
    lab = measure.label(bw.astype('int'), connectivity=connectivity)
    regions = measure.regionprops(lab, cache=False)
    areas = [f.area for f in regions]
    ai = np.argsort(areas)[::-1]
    bw = lab == ai[0] + 1
    return bw


def clean_seg(x, std=1):
    """
    Cleans a segmentation image.
    """

    # take out islands, fill in holes, and gaussian blur
    bw = extract_largest_vol(x)
    bw = 1 - extract_largest_vol(1 - bw)
    gadt = scipy.ndimage.gaussian_filter(bw.astype('float'), std)

    # figure out the proper threshold to maintain the total volume
    sgadt = np.sort(gadt.flatten())[::-1]
    thr = sgadt[np.ceil(bw.sum()).astype(int)]
    clean_bw = gadt > thr

    assert np.isclose(bw.sum(), clean_bw.sum(), atol=5), 'cleaning segmentation failed'
    return clean_bw.astype(float)


def clean_seg_batch(X_label, std=1):
    """
    Cleans batches of segmentation images.
    """
    if not X_label.dtype == 'float':
        X_label = X_label.astype('float')

    data = np.zeros(X_label.shape)
    for xi, x in enumerate(X_label):
        data[xi, ..., 0] = clean_seg(x[..., 0], std)

    return data


def filter_labels(atlas_vol, labels):
    """
    Filters given volumes to only include given labels, all other voxels are set to 0.
    """
    mask = np.zeros(atlas_vol.shape, 'bool')
    for label in labels:
        mask = np.logical_or(mask, atlas_vol == label)
    return atlas_vol * mask


def dist_trf(bwvol):
    """
    Computes positive distance transform from positive entries in a logical image.
    """
    revbwvol = np.logical_not(bwvol)
    return scipy.ndimage.morphology.distance_transform_edt(revbwvol)


def signed_dist_trf(bwvol):
    """
    Computes the signed distance transform from the surface between the binary
    elements of an image
    NOTE: The distance transform on either side of the surface will be +/- 1,
    so there are no voxels for which the distance should be 0.
    NOTE: Currently the function uses bwdist twice. If there is a quick way to
    compute the surface, bwdist could be used only once.
    """

    # get the positive transform (outside the positive island)
    posdst = dist_trf(bwvol)

    # get the negative transform (distance inside the island)
    notbwvol = np.logical_not(bwvol)
    negdst = dist_trf(notbwvol)

    # combine the positive and negative map
    return posdst * notbwvol - negdst * bwvol


def vol_to_sdt(X_label, sdt=True, sdt_vol_resize=1):
    """
    Computes the signed distance transform from a volume.
    """

    X_dt = signed_dist_trf(X_label)

    if not (sdt_vol_resize == 1):
        if not isinstance(sdt_vol_resize, (list, tuple)):
            sdt_vol_resize = [sdt_vol_resize] * X_dt.ndim
        if any([f != 1 for f in sdt_vol_resize]):
            X_dt = scipy.ndimage.interpolation.zoom(X_dt, sdt_vol_resize, order=1, mode='reflect')

    if not sdt:
        X_dt = np.abs(X_dt)

    return X_dt


def vol_to_sdt_batch(X_label, sdt=True, sdt_vol_resize=1):
    """
    Computes the signed distance transforms from volume batches.
    """

    # assume X_label is [batch_size, *vol_shape, 1]
    assert X_label.shape[-1] == 1, 'implemented assuming size is [batch_size, *vol_shape, 1]'
    X_lst = [f[..., 0] for f in X_label]  # get rows
    X_dt_lst = [vol_to_sdt(f, sdt=sdt, sdt_vol_resize=sdt_vol_resize)
                for f in X_lst]  # distance transform
    X_dt = np.stack(X_dt_lst, 0)[..., np.newaxis]
    return X_dt


def get_surface_pts_per_label(total_nb_surface_pts, layer_edge_ratios):
    """
    Gets the number of surface points per label, given the total number of surface points.
    """
    nb_surface_pts_sel = np.round(np.array(layer_edge_ratios) * total_nb_surface_pts).astype('int')
    nb_surface_pts_sel[-1] = total_nb_surface_pts - int(np.sum(nb_surface_pts_sel[:-1]))
    return nb_surface_pts_sel


def edge_to_surface_pts(X_edges, nb_surface_pts=None):
    """
    Converts edges to surface points.
    """

    # assumes X_edges is NOT in keras form
    surface_pts = np.stack(np.where(X_edges), 0).transpose()

    # random with replacements
    if nb_surface_pts is not None:
        chi = np.random.choice(range(surface_pts.shape[0]), size=nb_surface_pts)
        surface_pts = surface_pts[chi, :]

    return surface_pts


def sdt_to_surface_pts(X_sdt, nb_surface_pts,
                       surface_pts_upsample_factor=2, thr=0.50001, resize_fn=None):
    """
    Converts a signed distance transform to surface points.
    """
    us = [surface_pts_upsample_factor] * X_sdt.ndim

    if resize_fn is None:
        resized_vol = scipy.ndimage.interpolation.zoom(X_sdt, us, order=1, mode='reflect')
    else:
        resized_vol = resize_fn(X_sdt)
        pred_shape = np.array(X_sdt.shape) * surface_pts_upsample_factor
        assert np.array_equal(pred_shape, resized_vol.shape), 'resizing failed'

    X_edges = np.abs(resized_vol) < thr
    sf_pts = edge_to_surface_pts(X_edges, nb_surface_pts=nb_surface_pts)

    # can't just correct by surface_pts_upsample_factor because of how interpolation works...
    pt = [sf_pts[..., f] * (X_sdt.shape[f] - 1) / (X_edges.shape[f] - 1) for f in range(X_sdt.ndim)]
    return np.stack(pt, -1)


def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims], 
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def registration(m_mask, f_mask, m_image, m_label, method='affine'):
    assert method=='affine' or method=='rigid' or method=='nonrigid' or method=='bspline', '方法只有刚性、仿射和柔性'

    parameter_object = itk.ParameterObject.New()
    default_affine_parameter_map = parameter_object.GetDefaultParameterMap(method, 4)
    default_affine_parameter_map['FinalBSplineInterpolationOrder'] = ['0']
    # default_affine_parameter_map['MaximumNumberOfIterations'] = ['512']
    parameter_object.AddParameterMap(default_affine_parameter_map)    

    result, transform_parameters = itk.elastix_registration_method(
                        f_mask, m_mask, parameter_object=parameter_object, log_to_console=True)

    # warped_label = itk.transformix_filter(moving_label, transform_parameters)
    w_image = itk.transformix_filter(m_image, transform_parameters)
    w_label = itk.transformix_filter(m_label, transform_parameters)
    # w_mask = itk.transformix_filter(m_mask, transform_parameters)
    
    return w_image, w_label, result


def adjustWW(image, width=350, level=40):
    # 腹部窗宽350，窗位40
    v_min = level - (width / 2)
    v_max = level + (width / 2)

    img = image.copy()
    img[image < v_min] = v_min
    img[image > v_max] = v_max

    img = (img - v_min) / (v_max - v_min)
    # img = (img-img.mean()) / img.std()

    return img

def FFD_BraTS(brats_root, types=[]):
    
    for file in brats_root.rglob("*_seg.nii"):
        p_id = file.parent.name
        for tp in types:
        
            img_path = file.parent / (p_id + f'_{tp}.nii')
            # original_mesh_points = np.load('tests/test_datasets/meshpoints_sphere_orig.npy')
            img_itk = sitk.ReadImage(img_path)
            img_np = sitk.GetArrayFromImage(img_itk)

            lab_itk = sitk.ReadImage(file)
            lab_np = sitk.GetArrayFromImage(lab_itk)

            # new_img, new_lab = random_gen_flows(img_np, lab_np)
            new_img_ls = []
            new_lab_ls = []
            for i in range(img_np.shape[0]):
                img_slicer, lab_slicer = elastic_transform(img_np[i], lab_np[i], img_np.shape[1]*0.16, img_np.shape[1]*0.08, img_np.shape[1]*0.01)
                new_img_ls.append(img_slicer)
                new_lab_ls.append(lab_slicer)

            new_img = np.stack(new_img_ls)
            new_lab = np.stack(new_lab_ls)
            
            new_img_itk = sitk.GetImageFromArray(new_img.astype('int16'))
            new_img_itk.SetDirection(img_itk.GetDirection())
            new_img_itk.SetOrigin(img_itk.GetOrigin())
            new_img_itk.SetSpacing(img_itk.GetSpacing())

            new_lab_itk = sitk.GetImageFromArray(new_lab.astype('int16'))
            new_lab_itk.SetDirection(lab_itk.GetDirection())
            new_lab_itk.SetOrigin(lab_itk.GetOrigin())
            new_lab_itk.SetSpacing(lab_itk.GetSpacing())

            sitk.WriteImage(new_img_itk, file.parent / f'{file.parent.name}_{tp}1.nii')
            sitk.WriteImage(new_lab_itk, file.parent / f'{file.parent.name}_seg{tp}1.nii')
        print(f'{p_id} finished')


def FFD_one_file(img:np.array, seg:np.array):
    ffd = FFD()
    # ffd.read_parameters('tests/test_datasets/parameters_test_ffd_sphere.prm')
    nz, ny, nx = img.shape
    xv = np.linspace(0, nx, nx)
    yv = np.linspace(0, ny, ny)
    zv = np.linspace(0, nz, nz)
    z, y, x = np.meshgrid(zv, yv, xv)
    meshgrid = np.array([z.ravel(), y.ravel(), x.ravel()]) #3,z,y,x
    mesh = meshgrid.T # x,y,z,3

    new_locs =  torch.tensor(ffd(mesh).reshape((nx,ny,nz,3)).T).unsqueeze(0) 
    
    for i in range(len(img.shape)):
        new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (img.shape[i] - 1) - 0.5)
    new_locs = new_locs.permute(0, 2, 3, 4, 1)
    # new_locs = new_locs[..., [2, 1, 0]]
    
    new_img =  F.grid_sample(torch.tensor(img).double().unsqueeze(0).unsqueeze(0), torch.tensor(new_locs),  mode='bilinear', align_corners=False)
    new_lab =  F.grid_sample(torch.tensor(seg).double().unsqueeze(0).unsqueeze(0), torch.tensor(new_locs),  mode='bilinear', align_corners=False)

    return new_img[0,0].numpy(), new_lab[0,0].numpy()

def random_gen_flows(img:np.array, seg:np.array):
    stn = SpatialTransformer(size = img.shape)
    z, y, x = img.shape
    sizes = [z,y,x]
    sigma = 1

    flows = []
    for i in range(3):
        flow = np.random.normal(0, sigma, size=(sizes)) * (sizes[i]/(3*sigma)*np.random.random()*0.2) #z,y,w
        # flow = np.random.normal(0, sigma, size=(sizes)) #z,y,w
        flows.append(torch.tensor(gaussian_filter_3d(flow.transpose((2,1,0)), K_size=7, sigma=1)).permute(2,1,0))
    flows = torch.stack(flows).unsqueeze(0)

    # flows = [torch.randn(sizes) * size for size in sizes]
    # flows = torch.stack(flows).unsqueeze(0)
    
    new_img = stn(torch.tensor(img).float().unsqueeze(0).unsqueeze(0), flows)
    new_lab = stn(torch.tensor(seg).float().unsqueeze(0).unsqueeze(0), flows)

    return new_img[0,0].numpy(), new_lab[0,0].numpy()

def gaussian_filter_3d(img, K_size=3, sigma=1.5):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
 
    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float32)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float32)
 
    ## prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float32)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
 
    K /= (2 * np.pi * sigma * sigma)
 
    K /= K.sum()
    tmp = out.copy()
 
    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
 
    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    return out

def elastic_transform(image, label, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 6

    # pts1: 仿射变换前的点（3个点）
    pts1 = np.float32([center_square+square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square-square_size])

    # pts2: 仿射变换后的点
    # pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    pts2 = pts1 + np.random.normal(0, alpha_affine, size=pts1.shape).astype(np.float32)

    # # 仿射变换矩阵
    M = cv2.getAffineTransform(pts1, pts2)
    
    # # 对image进行仿射变换
    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101) 
    labelB = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # generate random displacement fields
    # random_state.rand(*shape)会产生一个和shape一样的服从[0,1]均匀分布的矩阵
    dx = gaussian_filter( (random_state.rand(*shape)*2-1), sigma)* alpha
    dy = gaussian_filter( (random_state.rand(*shape)*2-1), sigma)* alpha

    #generate meshgrid
    x, y, = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    indices = np.reshape(y+dy, (-1,1)), np.reshape(x+dx, (-1,1))

    # bilinear interpolation
    imageC = map_coordinates(imageB, indices, order=1, mode='constant').reshape(shape)
    labelC = map_coordinates(labelB, indices, order=1, mode='constant').reshape(shape)

    return imageC, labelC

if __name__ == '__main__':
    # create_grid(out_path='grid_pic.jpg')
    FFD_BraTS(Path('E:\\datasets\\BRAST2018\\MICCAI_BraTS_2018_Data_Training\\HGG'), types=['t1','t2','t1ce','flair'])