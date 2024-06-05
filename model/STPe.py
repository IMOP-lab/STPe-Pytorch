import timm
from collections import OrderedDict
import torch
import torch.nn as nn
from Model.base_models.resnet import resnet34
from torch.nn import functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax
class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()
        bn_momentum = 0.1
        def make_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels, momentum=bn_momentum),
                nn.ReLU()
            )
        self.enco1 = make_block(input_channels, 64)
        self.shortcut1 = nn.Conv2d(input_channels, 64, kernel_size=1, stride=1, bias=False)
        self.enco2 = make_block(64, 64)
        self.shortcut2 = nn.Identity() 
        self.enco3 = make_block(64, 128)
        self.shortcut3 = nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False)
        self.enco4 = make_block(128, 256)
        self.shortcut4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False)
        self.enco5 = make_block(256, 512)
        self.shortcut5 = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)
    def forward(self, x):
        identity = self.shortcut1(x)
        x = self.enco1(x) + identity
        a1 = F.max_pool2d(x, kernel_size=2, stride=2)
        identity = self.shortcut2(a1)
        x = self.enco2(a1) + identity
        a2 = F.max_pool2d(x, kernel_size=2, stride=2)
        identity = self.shortcut3(a2)
        x = self.enco3(a2) + identity
        a3 = F.max_pool2d(x, kernel_size=2, stride=2)
        identity = self.shortcut4(a3)
        x = self.enco4(a3) + identity
        a4 = F.max_pool2d(x, kernel_size=2, stride=2)
        identity = self.shortcut5(a4)
        x = self.enco5(a4) + identity
        a5 = F.max_pool2d(x, kernel_size=2, stride=2)
        return a1, a2, a3, a4, a5
class FFTAttentionModel(nn.Module):
    def __init__(self, channel):
        super(FFTAttentionModel, self).__init__()
        self.sk_attention = SKAttention(channel=6, kernels=[1, 3], reduction=8, L=16)
    def forward(self, x):
        x_fft = torch.fft.fftn(x, dim=(-2, -1))
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
        x_fft_real_imag = torch.cat((x_fft_shifted.real, x_fft_shifted.imag), dim=1)
        x_fft_attended = self.sk_attention(x_fft_real_imag)
        real_part = x_fft_attended[:, :x_fft_attended.size(1)//2, :, :]
        imag_part = x_fft_attended[:, x_fft_attended.size(1)//2:, :, :]
        x_fft_attended_complex = torch.complex(real_part, imag_part)
        x_ifft_shifted = torch.fft.ifftshift(x_fft_attended_complex, dim=(-2, -1))
        y_ifft = torch.fft.ifftn(x_ifft_shifted, dim=(-2, -1))
        y = y_ifft.real 
        return y
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        low_freq = x[:, :, 0::2, 0::2]
        high_freq = x[:, :, 1::2, 1::2]
        output = torch.cat((low_freq, high_freq), dim=1)  
        return output
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        channels //= 2
        low_freq = x[:, :channels, :, :]
        high_freq = x[:, channels:, :, :]
        output = torch.zeros(batch_size, channels, height*2, width*2).to(x.device)
        output[:, :, 0::2, 0::2] = low_freq
        output[:, :, 1::2, 1::2] = high_freq
        return output
class WaveletAttentionModel(nn.Module):
    def __init__(self, channel, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super(WaveletAttentionModel, self).__init__()
        self.dwt = DWT()
        self.attention = SKAttention(6, kernels=kernels, reduction=reduction, group=group, L=L) 
        self.iwt = IWT()
    def forward(self, x):
        dwt_result = self.dwt(x)
        attention_result = self.attention(dwt_result)
        output = self.iwt(attention_result)
        return output
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        return self.conv(x)
class SKAttention(nn.Module):
    def __init__(self, channel, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k//2, groups=group)),
                ('bn', nn.BatchNorm2d(channel)),
                ('relu', nn.ReLU())
            ])))
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)
    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)
        U = sum(conv_outs)
        S = U.mean(-1).mean(-1)
        Z = self.fc(S)
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))
        attention_weights = torch.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)
        V = (attention_weights * feats).sum(0)
        return V
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)  
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256, final_out_channels=None):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules) 
        self.final_out_channels = final_out_channels or in_channels
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(out_channels, self.final_out_channels, 1, bias=False),
            nn.BatchNorm2d(self.final_out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        self.apply(lambda m: m.eval() if isinstance(m, nn.BatchNorm2d) else None)
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
class PAM(Module): 
    def __init__(self, in_dim):
        super(PAM, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out
class CAM(Module): 
    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out
class ChannelHalvingLayer(nn.Module):
    def __init__(self, in_channels):
        super(ChannelHalvingLayer, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

    def forward(self, x):
        return self.conv1x1(x)
class gap(nn.Module):
    def __init__(self):
        super(gap, self).__init__()

    def forward(self, x):
        x_pool = torch.mean(x.view(x.size(0), x.size(1), x.size(2) * x.size(3)), dim=2)

        x_pool = x_pool.view(x.size(0), x.size(1), 1, 1).contiguous()
        return x_pool  
bn_momentum = 0.1 
class Elevate_Merge(nn.Module):
    def __init__(self, channels):
        super(Elevate_Merge, self).__init__()
        self.weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x1, x2):
        if x2.size()[2:] != x1.size()[2:]:
            x2 = self.upsample(x2)
        x = (self.weight[0] * x1 + self.weight[1] * x2) / (self.weight[0] + self.weight[1] + 1e-6)
        x = self.relu(x)
        return self.conv(x) 
class FeatureFusionModule(nn.Module):
    def __init__(self, channels_local, channels_global):
        super(FeatureFusionModule, self).__init__()
        self.local_conv = nn.Conv2d(channels_local, channels_global, 1)
        self.global_conv = nn.Conv2d(channels_global, channels_global, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, local_feature, global_feature):
        global_feature_upsampled = F.interpolate(global_feature, size=local_feature.shape[2:], mode='bilinear', align_corners=True)
        local_transformed = self.local_conv(local_feature)
        global_transformed = self.global_conv(global_feature_upsampled)
        fusion_weight = self.sigmoid(local_transformed + global_transformed)
        fused_feature = local_feature * fusion_weight + global_feature_upsampled * (1 - fusion_weight)
        return fused_feature
class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class DimensionMatchingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, target_size):
        super(DimensionMatchingLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.target_size = target_size
    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=True)
        return x 
class DynamicFusionModule(nn.Module):
    def __init__(self, channel1, channel2, out_channels):
        super(DynamicFusionModule, self).__init__()
        self.adjust_channel1 = nn.Conv2d(channel1, out_channels, kernel_size=1, stride=1, padding=0)
        self.adjust_channel2 = nn.Conv2d(channel2, out_channels, kernel_size=1, stride=1, padding=0)
        self.fusion_network = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid() 
        )
    def forward(self, feature1, feature2):
        feature1_upsampled = F.interpolate(feature1, scale_factor=2, mode='bilinear', align_corners=False)
        feature1_adjusted = self.adjust_channel1(feature1_upsampled)
        feature2_adjusted = self.adjust_channel2(feature2)
        combined_feature = torch.cat((feature1_adjusted, feature2_adjusted), dim=1)
        fusion_weight = self.fusion_network(combined_feature)
        fused_feature = fusion_weight * feature1_adjusted + (1 - fusion_weight) * feature2_adjusted
        return fused_feature 
class MS_CAM(nn.Module):
    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei   
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) 
        return x * scale
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out  
class nmcs(nn.Module):
    def __init__(self, in_channels,groups=2,):
        super().__init__()
        self.groups = groups
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, in_channels, 1, 1))
        self.cbias = Parameter(torch.ones(1, in_channels, 1, 1))        
        self.sweight = Parameter(torch.zeros(1, 1, 1, 1))
        self.sbias = Parameter(torch.ones(1, 1, 256, 256))
        self.sigmoid = nn.Sigmoid()
        self.conv_spatial = None        
    @staticmethod
    def channel_shuffle(x, groups):
        batch_size, num_channels, height, width = x.shape
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x
    def forward(self, x):
        x = self.channel_shuffle(x, self.groups)
        max_out_channel = self.max_pool(x)
        avg_out_channel = self.avg_pool(x)
        channel_attention = self.cweight * (max_out_channel + avg_out_channel) + self.cbias
        channel_attention = self.sigmoid(channel_attention)
        return x * channel_attention
GlobalAvgPool2D = lambda: nn.AdaptiveAvgPool2d(1)
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
class GLF(nn.Module):
    def __init__(self, in_channels, channel_in, out_channels, scale_aware_proj=False, r=4):
        super(GLF, self).__init__()
        self.scale_aware_proj = scale_aware_proj
        inter_channels = int(out_channels // r)
        self.scene_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )
        self.content_encoder = nn.Sequential(
            nn.Conv2d(channel_in, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.feature_reencoder = nn.Sequential(
            nn.Conv2d(channel_in, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(out_channels * 2, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, scene_feature, features):
        scene_feat = self.scene_encoder(scene_feature)
        content_feat = self.content_encoder(features)
        desired_height, desired_width = scene_feat.size()[2], scene_feat.size()[3]
        content_feat = nn.functional.interpolate(content_feat, size=(desired_height, desired_width), mode='bilinear', align_corners=False)
        combined_feat = torch.cat((scene_feat, content_feat), dim=1)  
        local_feat = self.local_att(combined_feat)
        global_feat = self.global_att(combined_feat)
        attention_feat = self.sigmoid(local_feat + global_feat) 
        reencoded_feature = self.feature_reencoder(features)
        enhanced_feature = reencoded_feature * attention_feat + reencoded_feature * (1 - attention_feat)
        return enhanced_feature
class AttentionBlock_attunet(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients, dilation_rate=2):
        super(AttentionBlock_attunet, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        if gate.size()[2:] != skip_connection.size()[2:]:
            gate = F.interpolate(gate, size=skip_connection.size()[2:], mode='bilinear', align_corners=True)
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return skip_connection * psi
class AttentionDecoder(nn.Module):
    def __init__(self, channel_list, attention_coefficients, final_out_channels=32, dilation_rate=2):       #
        super(AttentionDecoder, self).__init__()

        self.channel_conv = nn.ModuleList([
            nn.Conv2d(in_channels, final_out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate) for in_channels in channel_list
        ])

        self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.attention_blocks = nn.ModuleList([
            AttentionBlock_attunet(final_out_channels, final_out_channels, attention_coefficients[i], dilation_rate=dilation_rate)
            for i in range(len(channel_list))
        ])

    def forward(self, features):
        processed_features = []
        for i, feature in enumerate(features):
            feature = self.channel_conv[i](feature)
            if i > 0:  
                attention_feature = self.attention_blocks[i](feature, processed_features[i-1])
            else:
                attention_feature = feature
            attention_feature = self.upsample(attention_feature)
            processed_features.append(attention_feature)
        out_feat = sum(processed_features) / len(processed_features)
        return out_feat
class Conv_block(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d, scale=2, relu=True, last=False):
        super(Conv_block, self).__init__()
       
        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(in_planes, 2 * out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)     
        self.scale = scale
        self.last = last

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
    def forward(self, x):
        if self.last == False:
            x = self.conv_3x3(x)
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = self.conv_1x1(x)
        return x 
class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)
    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x
class STPe(nn.Module):
    def __init__(self,channel_list,attention_coefficients,num_class,input_channels):
    # def __init__(self, num_class):
        super(STPe,self).__init__()
        self.cbam=CBAM(512)
        self.backbone =resnet34(pretrained=True)    
        self.Elevate_Merge1 = Elevate_Merge(512)
        self.Elevate_Merge2 = Elevate_Merge(256)
        self.Elevate_Merge3 = Elevate_Merge(128)
        self.maxvit = timm.create_model(
          'maxvit_small_tf_224.in1k',
              pretrained=True,
              features_only=True,
         )  
        self.ms_cam=MS_CAM(512) 
        self.dim_match_layer1 = DimensionMatchingLayer(64, 64, (256, 256))
        self.dim_match_layer2 = DimensionMatchingLayer(96, 64, (128, 128))
        self.dim_match_layer3= DimensionMatchingLayer(192, 128, (64, 64))
        self.dim_match_layer4= DimensionMatchingLayer(384, 256, (32, 32))
        self.dim_match_layer5= DimensionMatchingLayer(768, 512, (16, 16))     
        self.fft_out = FFTAttentionModel(channel=24)
        self.wavelet_out=WaveletAttentionModel(channel=3)
        self.SCconv=ConvLayer(3,3)
        self.transformer_encoder = TransformerEncoder(d_model=512, nhead=8)
        self.dynamic_fusion_E4_E3 = DynamicFusionModule(channel1=512, channel2=256, out_channels=256)
        self.dynamic_fusion_E3_E2=DynamicFusionModule(channel1=256, channel2=128, out_channels=128)
        self.dynamic_fusion_E2_E1=DynamicFusionModule(channel1=128, channel2=128, out_channels=128)
        self.pam = PAM(512)  
        self.cam = CAM(512)
        self.feature_fusion = FeatureFusionModule(512,512) 
        self.se= SE(512)
        self.channel_halving_layer = ChannelHalvingLayer(1024)
        self.encoder = Encoder(input_channels=9)
        self.decoder = AttentionDecoder(
        channel_list=[320, 320, 640, 1280, 1024], 
        attention_coefficients=[32, 32,64, 128, 256]
         )
        out_planes = num_class
        self.aspp = ASPP(in_channels=512, atrous_rates=[6, 12, 18], out_channels=512) 
        self.nmcs1 = nmcs(in_channels=64)        
        self.nmcs2 = nmcs(in_channels=128)       
        self.nmcs3 = nmcs(in_channels=256)       
        self.nmcs4 = nmcs(in_channels=512)       
        self.Conv5=Conv_block(512,256,relu=False,last=True) 
        self.Conv4=Conv_block(256,128,relu=False) 
        self.Conv3=Conv_block(128,64,relu=False) 
        self.Conv2=Conv_block(64,64) 
        self.gap = GlobalAvgPool2D()
        self.global_local_fusion1 = GLF(512,512,512)
        self.global_local_fusion2 = GLF(512,512,256)
        self.global_local_fusion3= GLF(256,256,128)
        self.global_local_fusion4 = GLF(128,128,64)
        self.relu = nn.ReLU()  
        self.softmax = nn.Softmax(dim=1)
        self.finall_conv= nn.Conv2d(32, out_planes, 1)
        self.upsampling= nn.UpsamplingBilinear2d(scale_factor=2)
        self.channel_mapping = nn.Sequential(
                    nn.Conv2d(1024, out_planes, 3,1,1), 
                    nn.BatchNorm2d(out_planes),
                    nn.ReLU(True)
                )
        self.direc_reencode = nn.Sequential(
                    nn.Conv2d(out_planes, out_planes, 1),
                )
    def forward(self, x):
        y = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        maxvit_features = self.maxvit(y)
        e1, e2, e3, e4, e5 = maxvit_features[:5]        
        e1 = self.dim_match_layer1(e1)
        e2 = self.dim_match_layer2(e2)
        e3=  self.dim_match_layer3(e3)
        e4=  self.dim_match_layer4(e4)
        e5=  self.dim_match_layer5(e5)
        e5=self.aspp(e5)  
        Conv_x=self.SCconv(x)
        x_fft=self.fft_out(x)
        x_wavelet=self.wavelet_out(x)
        x_combined =  torch.cat([Conv_x, x_fft, x_wavelet], dim=1)
        a1,a2,a3,a4,a5 = self.encoder(x_combined)       
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)  
        c1 = self.backbone.relu(x)
        x = self.backbone.maxpool(c1)  
        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        c11 = torch.cat([e1, c1], dim=1) 
        c22 = torch.cat([e2, c2], dim=1)
        c33= torch.cat([e3, c3], dim=1)
        c44 = torch.cat([e4, c4], dim=1)
        c55 = torch.cat([e5, c5], dim=1)
        c55_halve= self.channel_halving_layer(c55)
        b, c, h, w = c55_halve.size()
        c555_flat = c55_halve.view(b, c, h * w).permute(0, 2, 1)
        transformer_output = self.transformer_encoder(c555_flat)
        transformer_output = transformer_output.permute(0, 2, 1).view(b, c, h, w)
        global_feature = self.gap(transformer_output)    
        c55_pam = self.pam(c55_halve)
        c55_cam = self.cam(c55_halve) 
        c55sum=c55_pam+c55_cam 
        fused_feature= self.feature_fusion(c55sum, global_feature)
        c5_enhance = self.se(fused_feature)
        a1=self.nmcs1(a1)
        a_1 = torch.cat([a1, c11], dim=1)
        a2=self.nmcs1(a2)
        a_2 = torch.cat([a2,c22], dim=1)
        a3=self.nmcs2(a3)
        a_3 = torch.cat([a3, c33], dim=1)
        a4=self.nmcs3(a4)
        a_4 = torch.cat([a4, c44], dim=1)
        a5=self.ms_cam(a5)
        a5=self.nmcs4(a5)
        b, c, h, w = a5.size()
        a5_flat = a5.view(b, c, h * w).permute(0, 2, 1)
        transformer_output = self.transformer_encoder(a5_flat)
        transformer_output = transformer_output.permute(0, 2, 1).view(b, c, h, w)
        global_feature = self.gap(transformer_output)    
        a5_pam = self.pam(a5)
        a5_cam = self.cam(a5) 
        a5sum=a5_pam+a5_cam 
        fused_feature = self.feature_fusion(a5sum, global_feature)
        a_5 = self.se(fused_feature) 
        c5=self.cbam(c5_enhance)
        H5 = self.global_local_fusion1(self.gap(c5), c5)
        E4 = self.relu(self.Conv5(H5) + c44)
        E4 = self.Elevate_Merge1(E4, H5) 
        H4 = self.global_local_fusion2(self.gap(H5), E4)
        E3 = self.relu(self.Conv4(H4) + c33)
        E3 = self.Elevate_Merge2(E3, H4)
        E3 = self.dynamic_fusion_E4_E3(E4, E3)
        H3 = self.global_local_fusion3(self.gap(H4), E3)
        E2 = self.relu(self.Conv3(H3) + c22)
        E2 = self.Elevate_Merge3(E2, H3) 
        E2 = self.dynamic_fusion_E3_E2(E3, E2)
        H2 = self.global_local_fusion4(self.gap(H3), E2)
        E1 = self.relu(self.Conv2(H2) + c11) 
        E1 = self.dynamic_fusion_E2_E1(E2, E1)
        E1 = torch.cat([F.interpolate(a_1, size=E1.shape[2:]), E1], dim=1)
        E2 = torch.cat([F.interpolate(a_2, size=E2.shape[2:]), E2], dim=1)
        E3 = torch.cat([F.interpolate(a_3, size=E3.shape[2:]), E3], dim=1)
        E4 = torch.cat([F.interpolate(a_4, size=E4.shape[2:]), E4], dim=1)
        c5 = torch.cat([F.interpolate(a_5, size=c5.shape[2:]), c5], dim=1) 
        feat_list = [E1,E2,E3,E4,c5]  
        Final_feature = self.decoder(feat_list)   
        SPTe_pred = self.finall_conv(Final_feature)  
        SPTe_pred=self.upsampling(SPTe_pred)
        SPTe_pred= self.softmax(SPTe_pred) 
        return SPTe_pred   



