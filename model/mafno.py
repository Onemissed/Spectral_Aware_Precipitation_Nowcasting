import torch
import math
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

class BasicConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True,
                 enc=False):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        # PixelShuffle used for upsampling
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
            self.norm = nn.GroupNorm(2, out_channels)
        elif enc is True:
            # Inception encoder
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3,
                stride=stride, padding=padding, dilation=dilation)
            self.conv2 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, padding=0, dilation=dilation)
            self.conv3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, padding=0, dilation=dilation))
            self.norm1 = nn.GroupNorm(2, out_channels)
            self.norm2 = nn.GroupNorm(2, out_channels)
            self.norm3 = nn.GroupNorm(2, out_channels)
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)
            self.norm = nn.GroupNorm(2, out_channels)

        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)
        self.upsampling = upsampling
        self.enc = enc

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.enc:
            x1 = self.conv1(x)
            x2 = self.conv2(x)
            x3 = self.conv3(x)
            if self.act_norm:
                x1 = self.act(self.norm1(x1))
                x2 = self.act(self.norm2(x2))
                x3 = self.act(self.norm3(x3))
            # Concatenation
            y = torch.cat([x1, x2, x3], dim=1)
        else:
            y = self.conv(x)
            if self.act_norm:
                y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):
    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True,
                 enc=False):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace, enc=enc)

    def forward(self, x):
        T, C, H, W = x.size()
        y = self.conv(x)

        return y


class mAFNO_model(nn.Module):
    def __init__(self, args, in_shape, hid_S=16, N_S=4, N_T=4,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, **kwargs):
        super(mAFNO_model, self).__init__()
        T, C, H, W = in_shape
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))
        act_inplace = False
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(T, args.output_length, hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        self.hid = MidMetaNet(T, T * hid_S, args.output_length * hid_S, N_T,
                              input_resolution=(H, W),
                              mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        self.out_T = args.output_length

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.contiguous().view(B * T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)

        hid = self.hid(z)
        hid = hid.reshape(B * self.out_T, C_, H_, W_)

        Y = self.dec(hid, skip)

        Y = Y.reshape(B, self.out_T, C, H, W)

        return Y

def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]

class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace, enc=True),
            *[ConvSC(3 * C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace, enc=True) for s in samplings[1:]]
        )
        self.fuse1 = nn.Conv2d(3 * C_hid, C_hid, 1)
        self.fuse2 = nn.Conv2d(3 * C_hid, C_hid, 1)

    def forward(self, x):
        T, C, H, W = x.size()
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        latent = self.fuse1(latent)
        enc1 = self.fuse2(enc1)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, in_T, out_T, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
        self.in_T = in_T
        self.out_T = out_T

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)

        B, C, H, W = enc1.shape
        enc1 = enc1.reshape(B // self.in_T, self.in_T, C, H, W)
        # Select the last frame of the shallow encoder feature and copy it T' times as the residual
        enc1 = enc1[:, -1:].repeat(1, self.out_T, 1, 1, 1)
        enc1 = enc1.reshape(-1, C, H, W)

        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class MetaBlock(nn.Module):
    def __init__(self, T, in_channels, out_channels, input_resolution=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = AFNOBlock(T, in_channels, out_channels, spatio_kernel=3, embed_dim=out_channels, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path=drop_path)

    def forward(self, x):
        z = self.block(x)
        return z


class MidMetaNet(nn.Module):
    def __init__(self, T, channel_in, channel_hid, N2,
                 input_resolution=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        enc_layers = [MetaBlock(T,
            channel_in, channel_hid, input_resolution,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2):
            enc_layers.append(MetaBlock(T,
                channel_hid, channel_hid, input_resolution,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))

        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, -1, C, H, W)
        return y


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Spatial attention in parallel convolution branch
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pooled = torch.cat([max_pool, avg_pool], dim=1)   # [B, 2, H, W]
        attn = self.conv(pooled)
        attn = self.sigmoid(attn)
        return x * attn


class AFNO_SubBlock(nn.Module):
    def __init__(self, T, C_in, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_fno=False, use_blocks=False):
        super().__init__()
        self.norm1 = norm_layer(C_in)

        self.filter = AFNO2D(T=T, C_in=C_in, dim=dim, hidden_size=dim, num_blocks=1, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.double_skip = True

        self.input_dim = C_in
        self.hidden_dim = dim
        self.proj = nn.Linear(C_in, dim)

        if self.input_dim == self.hidden_dim:
            # The CNN branch
            self.batch_norm = nn.BatchNorm2d(C_in)
            self.depthwise = nn.Conv2d(C_in, dim, kernel_size=3, padding=1, stride=1, groups=dim, bias=True)
            self.pointwise = nn.Conv2d(C_in, dim, kernel_size=1, bias=True)
            self.sa = SpatialAttention(kernel_size=5)
            self.fuse_mlp = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
            )

    def forward(self, x):
        residual = x

        if self.input_dim != self.hidden_dim:
            residual = self.proj(residual)

        if self.input_dim == self.hidden_dim:
            # use the CNN branch with spatial attention to extract static local features
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x_spatial = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
            feature_spatial = F.gelu(self.depthwise(self.batch_norm(x_spatial)))
            feature_spatial = self.pointwise(feature_spatial)
            feature_spatial = self.sa(feature_spatial)
            feature_spatial = feature_spatial.permute(0, 2, 3, 1).reshape(B, N, -1)

        # Layer normalization
        x = self.norm1(x)
        # mAFNO filter
        x = self.filter(x)

        if self.input_dim == self.hidden_dim:
            merged = torch.cat([x, feature_spatial], dim=-1)
            fused = self.fuse_mlp(merged)  # shape: [B, N, C]
            x = self.drop_path(fused) + residual
        else:
            x = self.drop_path(x) + residual

        residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class AFNO2D(nn.Module):
    def __init__(self, T, C_in, dim, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, C_in, self.block_size))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

        self.T = T

        num_center = 11
        self.centers = nn.Parameter(torch.linspace(-1, 1, num_center), requires_grad=False)

        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1, 1, self.hidden_size, num_center))
        self.b3 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x, spatial_size=None):
        # No residual inside the AFNO
        # bias = x

        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape

        if spatial_size == None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size

        x = x.reshape(B, H, W, C)
        # use fft2 instead of rfft2
        x = torch.fft.fft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, C)

        o1_real = (
            torch.einsum('...bi,bio->...bo', x.real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x.imag, self.w1[1]) + \
            self.b1[0]
        )
        o1_imag = (
            torch.einsum('...bi,bio->...bo', x.imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x.real, self.w1[1]) + \
            self.b1[1]
        )

        # Multi-bias ReLU activation
        temp_real1 = o1_real.unsqueeze(-1) - self.centers
        temp_real2 = F.relu(temp_real1)

        temp_imag1 = o1_imag.unsqueeze(-1) - self.centers
        temp_imag2 = F.relu(temp_imag1)

        temp_real3 = temp_real2 * self.weight[0] - temp_imag2 * self.weight[1]
        temp_real3 = torch.sum(temp_real3, dim=5) + self.b3[0]

        temp_imag3 = temp_real2 * self.weight[1] + temp_imag2 * self.weight[0]
        temp_imag3 = torch.sum(temp_imag3, dim=5) + self.b3[1]

        o2_real = (
            torch.einsum('...bi,bio->...bo', temp_real3, self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', temp_imag3, self.w2[1]) + \
            self.b2[0]
        )
        o2_imag = (
            torch.einsum('...bi,bio->...bo', temp_imag3, self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', temp_real3, self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)

        x = torch.fft.ifft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.reshape(B, N, -1)
        x = x.real
        x = x.type(dtype)

        return x


class AFNOBlock(nn.Module):
    def __init__(self, T, C_in, C_hid, spatio_kernel, embed_dim, mlp_ratio, drop_rate, drop_path, norm_layer=nn.LayerNorm, use_fno=False, use_blocks=False, act_inplace=True):
        super(AFNOBlock, self).__init__()

        self.forward_feature = AFNO_SubBlock(T=T, C_in=C_in, dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=drop_path, norm_layer=norm_layer, use_fno=use_fno, use_blocks=use_blocks)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.forward_feature(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return x