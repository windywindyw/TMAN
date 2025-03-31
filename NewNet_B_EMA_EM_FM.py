import torch
import torch.nn as nn
from torchvision import models
from torch.jit import Final
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class FusionAttention(nn.Module):
    def __init__(self, in_channels, out_channels=768):
        super(FusionAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

        self.conv4 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)

        self.msa = nn.MultiheadAttention(embed_dim=in_channels // 2, num_heads=8)

        self.output_conv = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.f_ch_q = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, 1, bias=False)
        )

    def forward(self, F1, F2, F3):
        B, C, H1, W1 = F1.shape
        B, C, H2, W2 = F2.shape
        B, C, H3, W3 = F3.shape

        F1_proj = self.f_ch_q(self.avg_pool(F1)) + self.f_ch_q(self.max_pool(F1))
        F1_proj = self.conv1(F1_proj).view(B, -1, 1).permute(2, 0, 1)  # Shape: (H1*W1, B, C2)

        F2_proj = self.f_ch_q(self.avg_pool(F2)) + self.f_ch_q(self.max_pool(F2))
        F2_proj = self.conv2(F2_proj).view(B, -1, 1).permute(2, 0, 1)  # Shape: (H2*W2, B, C2)

        F3_proj = self.conv3(F3).view(B, -1, H3 * W3).permute(2, 0, 1)  # Shape: (H2*W2, B, C2)

        # Multi-Scale Attention
        F1_msa, _ = self.msa(F1_proj, F3_proj, F3_proj)
        F2_msa, _ = self.msa(F2_proj, F3_proj, F3_proj)

        F1 = self.conv1(F1).view(B, -1, H1 * W1).permute(2, 0, 1)

        F3_msa, _ = self.msa(F1, F1_msa, F2_msa)
        F3_msa = F3_msa.permute(1, 2, 0).view(B, -1, H1, W1)  # Reshape back to (B, C2, H2, W2)

        F2_out = self.output_conv(F3_msa)

        return F2_out


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class AttMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bias=True):
        # channel last
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    fast_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        # assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = max(dim // num_heads, 32)
        self.scale = self.head_dim ** -0.5
        self.fast_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  # FIXME

        self.qkv = nn.Linear(dim, self.num_heads * self.head_dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fast_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.head_dim * self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class AttentionBlock(nn.Module):

    def __init__(
            self,
            dim, mlp_ratio=4., num_heads=8, qkv_bias=False, qk_norm=False, drop=0., attn_drop=0.,
            init_values=None, drop_path=0.2, act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = AttMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.size()
        x = x.reshape(B, H * W, C).contiguous()

        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        x = x.reshape(B, H, W, C).contiguous()
        return x


class BreastBiomarkerNet(nn.Module):
    def __init__(self, num_classes=8):
        super(BreastBiomarkerNet, self).__init__()
        self.backbone = models.convnext_tiny(weights="DEFAULT")
        self.backbone.classifier[2] = nn.Linear(768, num_classes)

        # Add EMA module
        self.ema = EMA(768)
        self.attblock = AttentionBlock(768)
        self.fusionblock = FusionAttention(768)

    def forward(self, x):
        x = self.backbone.features(x)

        x2 = self.ema(x)
        x3 = self.attblock(x)
        x3 = x3.permute(0, 3, 1, 2)

        x = x + self.fusionblock(x, x2, x3)

        x = self.backbone.avgpool(x)
        x = self.backbone.classifier(x)

        return x
