import torch
import torch.nn as nn
from torch.nn import ReLU, LeakyReLU
import torch.nn.functional as F
from torch.distributions.normal import Normal

def default_unet_features():
    nb_features = [[16, 32, 32, 32, 32], [32, 32, 32, 32, 16, 16]]  # encoder  # decoder
    return nb_features


def conv(dim=3):
    if dim == 2:
        return nn.Conv2d
    return nn.Conv3d

def trans_conv(dim=3):
    if dim == 2:
        return nn.ConvTranspose2d
    return nn.ConvTranspose3d


def convolve(in_channels, out_channels, kernel_size, stride, dim=3):
    return conv(dim=dim)(in_channels, out_channels, kernel_size, stride=stride, padding=1)


def convolveReLU(in_channels, out_channels, kernel_size, stride, dim=3):
    return nn.Sequential(ReLU, convolve(in_channels, out_channels, kernel_size, stride, dim=dim))


def convolveLeakyReLU(in_channels, out_channels, kernel_size, stride, dim=3):
    return nn.Sequential(LeakyReLU(0.1), convolve(in_channels, out_channels, kernel_size, stride, dim=dim))


def upconvolve(in_channels, out_channels, kernel_size, stride, dim=3):
    return trans_conv(dim=dim)(in_channels, out_channels, kernel_size, stride, padding=1)


def upconvolveReLU(in_channels, out_channels, kernel_size, stride, dim=3):
    return nn.Sequential(ReLU, upconvolve(in_channels, out_channels, kernel_size, stride, dim=dim))


def upconvolveLeakyReLU(in_channels, out_channels, kernel_size, stride, dim=3):
    return nn.Sequential(LeakyReLU(0.1), upconvolve(in_channels, out_channels, kernel_size, stride, dim=dim))


class VTN(nn.Module):
    def __init__(self, dim=3, flow_multiplier=1., channels=16):
        super(VTN, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.dim = dim

        # Network architecture
        # The first convolution's input is the concatenated image
        self.conv1 = convolveLeakyReLU(2, channels, 3, 2, dim=dim)
        self.conv2 = convolveLeakyReLU(channels, 2 * channels, 3, 2, dim=dim)
        self.conv3 = convolveLeakyReLU(2 * channels, 4 * channels, 3, 2, dim=dim)
        self.conv3_1 = convolveLeakyReLU(4 * channels, 4 * channels, 3, 1, dim=dim)
        self.conv4 = convolveLeakyReLU(4 * channels, 8 * channels, 3, 2, dim=dim)
        self.conv4_1 = convolveLeakyReLU(8 * channels, 8 * channels, 3, 1, dim=dim)
        self.conv5 = convolveLeakyReLU(8 * channels, 16 * channels, 3, 2, dim=dim)
        self.conv5_1 = convolveLeakyReLU(16 * channels, 16 * channels, 3, 1, dim=dim)
        # self.conv6 = convolveLeakyReLU(16 * channels, 32 * channels, 3, 2, dim=dim)
        # self.conv6_1 = convolveLeakyReLU(32 * channels, 32 * channels, 3, 1, dim=dim)

        # self.pred6 = convolve(32 * channels, dim, 3, 1, dim=dim)
        # self.upsamp6to5 = upconvolve(dim, dim, 4, 2, dim=dim)
        # self.deconv5 = upconvolveLeakyReLU(32 * channels, 16 * channels, 4, 2, dim=dim)

        # self.pred5 = convolve(32 * channels + dim, dim, 3, 1, dim=dim)  # 514 = 32 * channels + 1 + 1
        self.pred5 = convolve(16 * channels, dim, 3, 1, dim=dim)  # 514 = 32 * channels + 1 + 1
        self.upsamp5to4 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv4 = upconvolveLeakyReLU(16 * channels, 8 * channels, 4, 2, dim=dim)
        # self.deconv4 = upconvolveLeakyReLU(32 * channels + dim, 8 * channels, 4, 2, dim=dim)

        self.pred4 = convolve(16 * channels + dim, dim, 3, 1, dim=dim)  # 258 = 64 * channels + 1 + 1
        self.upsamp4to3 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv3 = upconvolveLeakyReLU(16 * channels + dim,  4 * channels, 4, 2, dim=dim)

        self.pred3 = convolve(8 * channels + dim, dim, 3, 1, dim=dim)
        self.upsamp3to2 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv2 = upconvolveLeakyReLU(8 * channels + dim, 2 * channels, 4, 2, dim=dim)

        self.pred2 = convolve(4 * channels + dim, dim, 3, 1, dim=dim)
        self.upsamp2to1 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv1 = upconvolveLeakyReLU(4 * channels + dim, channels, 4, 2, dim=dim)

        self.pred0 = upconvolve(2 * channels + dim, dim, 4, 2, dim=dim)

    def forward(self, moving, fixed ):
        concat_image = torch.cat((fixed, moving), dim=1)  # 2 x 512 x 512
        x1 = self.conv1(concat_image)  # 16 x 256 x 256
        x2 = self.conv2(x1)  # 32 x 128 x 128
        x3 = self.conv3(x2)  # 1 x 64 x 64 x 64
        x3_1 = self.conv3_1(x3)  # 64 x 64 x 64
        x4 = self.conv4(x3_1)  # 128 x 32 x 32
        x4_1 = self.conv4_1(x4)  # 128 x 32 x 32
        x5 = self.conv5(x4_1)  # 256 x 16 x 16
        x5_1 = self.conv5_1(x5)  # 256 x 16 x 16

        # pred6 = self.pred6(x5_1)  # 2 x 8 x 8
        # upsamp6to5 = self.upsamp6to5(pred6)  # 2 x 16 x 16
        # deconv5 = self.deconv5(x5_1)  # 256 x 16 x 16
        # concat5 = torch.cat([x5_1, deconv5, upsamp6to5], dim=1)  # 514 x 16 x 16
      
        pred5 = self.pred5(x5_1)  # 2 x 16 x 16
        upsamp5to4 = self.upsamp5to4(pred5)  # 2 x 32 x 32
        deconv4 = self.deconv4(x5_1)  # 2 x 32 x 32
        concat4 = torch.cat([x4_1, deconv4, upsamp5to4], dim=1)  # 258 x 32 x 32

        pred4 = self.pred4(concat4)  # 2 x 32 x 32
        upsamp4to3 = self.upsamp4to3(pred4)  # 2 x 64 x 64
        deconv3 = self.deconv3(concat4)  # 64 x 64 x 64
        concat3 = torch.cat([x3_1, deconv3, upsamp4to3], dim=1)  # 130 x 64 x 64

        pred3 = self.pred3(concat3)  # 2 x 63 x 64
        upsamp3to2 = self.upsamp3to2(pred3)  # 2 x 128 x 128
        deconv2 = self.deconv2(concat3)  # 32 x 128 x 128
        concat2 = torch.cat([x2, deconv2, upsamp3to2], dim=1)  # 66 x 128 x 128

        pred2 = self.pred2(concat2)  # 2 x 128 x 128
        upsamp2to1 = self.upsamp2to1(pred2)  # 2 x 256 x 256
        deconv1 = self.deconv1(concat2)  # 16 x 256 x 256
        concat1 = torch.cat([x1, deconv1, upsamp2to1], dim=1)  # 34 x 256 x 256

        pred0 = self.pred0(concat1)  # 2 x 512 x 512

        return pred0 * 20 * self.flow_multiplier  # why the 20?


class VTNAffineStem(nn.Module):

    def __init__(self, dim=1, channels=16, flow_multiplier=1., im_size=(64,128,128), id_map=None):
        super(VTNAffineStem, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.dim = dim
        self.id_map = id_map

        # Network architecture
        # The first convolution's input is the concatenated image
        self.conv1 = convolveLeakyReLU(2, channels, 3, 2, dim=self.dim)
        self.conv2 = convolveLeakyReLU(channels, 2 * channels, 3, 2, dim=dim)
        self.conv3 = convolveLeakyReLU(2 * channels, 4 * channels, 3, 2, dim=dim)
        self.conv3_1 = convolveLeakyReLU(4 * channels, 4 * channels, 3, 1, dim=dim)
        self.conv4 = convolveLeakyReLU(4 * channels, 8 * channels, 3, 2, dim=dim)
        self.conv4_1 = convolveLeakyReLU(8 * channels, 8 * channels, 3, 1, dim=dim)
        self.conv5 = convolveLeakyReLU(8 * channels, 16 * channels, 3, 2, dim=dim)
        self.conv5_1 = convolveLeakyReLU(16 * channels, 16 * channels, 3, 1, dim=dim)
        self.conv6 = convolveLeakyReLU(16 * channels, 32 * channels, 3, 2, dim=dim)
        self.conv6_1 = convolveLeakyReLU(32 * channels, 32 * channels, 3, 1, dim=dim)

        # I'm assuming that the image's shape is like (im_size, im_size, im_size)
        self.last_conv_size = torch.tensor(im_size).div( (self.channels * 4)).ceil()

        self.fc_loc = nn.Sequential(
            nn.Linear(512 * self.last_conv_size.prod().int(), 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 6*(dim - 1))
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        """
        Identity Matrix
            | 1 0 0 0 |
        I = | 0 1 0 0 | 
            | 0 0 1 0 |
        """
        if dim == 3:
            self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))
        else:
            self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def forward(self, moving, fixed ):
        concat_image = torch.cat((fixed, moving), dim=1)  # 2 x 512 x 512
        x1 = self.conv1(concat_image)  # 16 x 256 x 256
        x2 = self.conv2(x1)  # 32 x 128 x 128
        x3 = self.conv3(x2)  # 1 x 64 x 64 x 64
        x3_1 = self.conv3_1(x3)  # 64 x 64 x 64
        x4 = self.conv4(x3_1)  # 128 x 32 x 32
        x4_1 = self.conv4_1(x4)  # 128 x 32 x 32
        x5 = self.conv5(x4_1)  # 256 x 16 x 16
        x5_1 = self.conv5_1(x5)  # 256 x 16 x 16
        x6 = self.conv6(x5_1)  # 512 x 8 x 8
        x6_1 = self.conv6_1(x6)  # 512 x 8 x 8

        # Affine transformation
        xs = x6_1.view(-1, 512 * self.last_conv_size.prod().int())
        # xs = x6_1.view(-1, 512 * 1 * 1 * 1)

        if self.dim == 3:
            theta = self.fc_loc(xs).view(-1, 3, 4)
        else:
            theta = self.fc_loc(xs).view(-1, 2, 3)
        
        flow = F.affine_grid(theta, moving.size(), align_corners=False)  # batch x 512 x 512 x 2

        if self.dim == 2:
            flow = flow.permute(0, 3, 1, 2)  # batch x 2 x 512 x 512
        else:
             flow = flow.permute(0, 4, 1, 2, 3)
        
        if self.id_map is not None:
            id_loss = theta - self.id_map
            return flow, torch.mean(torch.square(id_loss))
        else:
            return flow

class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = 3

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError(
                    "must provide unet nb_levels if nb_features is an integer"
                )
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(
                int
            )
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError("cannot use nb_levels if nb_features is not an integer")
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            if prev_nf == 2:
                self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            else:
                self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[: len(self.enc_nf) - 1]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 16
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf) - 1 :]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))
        x = x_enc[-1]
        # conv, upsample, concatenate series
        x = x_enc.pop()
        i = 0
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x

class VxmDense(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(
        self,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = 3
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # configure core unet model
        self.unet_model = Unet(
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, "Conv%dd" % ndims)
        self.flow1 = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)
        self.flow2 = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)
        # init flow layer with small weights and bias
        self.flow1.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow1.weight.shape))
        self.flow2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow2.weight.shape))
        self.flow1.bias = nn.Parameter(torch.zeros(self.flow1.bias.shape))
        self.flow2.bias = nn.Parameter(torch.zeros(self.flow2.bias.shape))


    def forward(self, source, target, registration=True):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        """

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field1 = self.flow1(x)
        flow_field2 = self.flow2(x)

        # return flow_field1,flow_field2
        return flow_field1
    
class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, "Conv%dd" % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

if __name__ == "__main__":
    vtn_model = VTN(dim=3)
    vtn_affine_model = VTNAffineStem(dim=3, im_size=512)
    x = torch.randn(1, 512, 1, 512, 512)
    y1 = vtn_model(x, x)
    y2 = vtn_affine_model(x, x)

    assert  y1.size() == y2.size()
