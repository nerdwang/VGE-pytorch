import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

#**************************************************************总模型***********************************************************************************#
class VGE(nn.Module):
    def __init__(self, eye_channels=1, gmap_channels=2, filters_channels=64):
        super().__init__()
        self.prior_enc = prior_enc(eye_channels)
        self.post_enc = post_enc(eye_channels, gmap_channels)
        self.latent_encoder = latent_encoder()
        self.dec = dec()
        self.build_after = build_after(filters_channels)
        self.densenet = densenet(2)

    def forward(self, batch):
        x, _, y1 = batch
        gs = self.prior_enc(x)
        zs_prior, m_v_prior = self.latent_encoder(gs)
        if self.training:  # Check if the model is in training mode
            hs = self.post_enc(x, y1)
            _, m_v_post = self.latent_encoder(hs)
        else:
            m_v_post = None
        last_dec = self.dec(gs, zs_prior, m_v_prior, sample=[True, True, True, True, True, True, True])
        gmap = self.build_after(last_dec)
        gaze_direction = self.densenet(gmap)
        return gaze_direction, gmap, m_v_post, m_v_prior
    


 #***************************************************************CVAE部分*****************************************************************************#
class WeightNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3, stride=1, padding='SAME', init_scale=1.0, scale=False,initial=False):
        super(WeightNormConv2d, self).__init__()
        self.V = nn.Parameter(torch.Tensor(out_channels, in_channels, filter_size, filter_size))
        self.g = nn.Parameter(torch.Tensor(out_channels))
        self.b = nn.Parameter(torch.Tensor(out_channels))
        self.stride = stride
        if padding == 'SAME':
           self.padding = (filter_size - 1) // 2
        self.scale = scale
        if self.scale:
            self.rezero_scale = nn.Parameter(torch.tensor(0.0))

        init.normal_(self.V, mean=0, std=0.05)
        init.constant_(self.g, 1.)
        init.constant_(self.b, 0.)

        self.init_scale = init_scale
        self.initial = initial

    def forward(self, x):
        V_norm = F.normalize(self.V.view(self.V.size(0), -1), dim=1).view_as(self.V)
        x = F.conv2d(x, V_norm, None, self.stride, self.padding)
        if self.initial:
            mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            var = torch.var(x, dim=(0, 2, 3), keepdim=True)
            self.g.data = self.init_scale / torch.sqrt(var + 1e-10)
            self.b.data = -mean * self.g.data
        x = x * self.g.view(1, -1, 1, 1) + self.b.view(1, -1, 1, 1)
        if self.scale:
            x = self.rezero_scale * x
        return x
        

class WeightNormConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3, stride=1, padding='SAME', init_scale=1.0, initial=False):
        super(WeightNormConvTranspose2d, self).__init__()
        self.V = nn.Parameter(torch.Tensor(in_channels, out_channels, filter_size, filter_size))
        self.g = nn.Parameter(torch.Tensor(out_channels))
        self.b = nn.Parameter(torch.Tensor(out_channels))
        self.stride = stride
        self.padding = padding
        self.filter_size = filter_size

        init.normal_(self.V, mean=0, std=0.05)
        init.constant_(self.g, 1.)
        init.constant_(self.b, 0.)

        self.init_scale = init_scale
        self.initial = initial

    def forward(self, x):
        V_norm = F.normalize(self.V.view(self.V.size(1), -1), dim=1).view_as(self.V)
        if self.padding == 'SAME':
            input_height, input_width = x.size(2), x.size(3)
            output_height = input_height * self.stride
            output_width = input_width * self.stride
            output_size = (x.size(0), self.V.size(1), output_height, output_width)
        else:
            input_height, input_width = x.size(2), x.size(3)
            output_height = input_height * self.stride + self.filter_size - 1
            output_width = input_width * self.stride + self.filter_size - 1
            output_size = (x.size(0), self.V.size(1), output_height, output_width)

        x = F.conv_transpose2d(x, V_norm, None, self.stride, self.padding, output_size=output_size)
        if not self.initial:
            mean, var = torch.mean(x, dim=(0, 2, 3), keepdim=True), torch.var(x, dim=(0, 2, 3), keepdim=True)
            self.g.data = self.init_scale / torch.sqrt(var + 1e-10)
            self.b.data = -mean.squeeze() * self.g.data
        x = x * self.g.view(1, -1, 1, 1) + self.b.view(1, -1, 1, 1)
        return x


    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, a_channels=None, use_batch_statistics=True, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.use_batch_statistics = use_batch_statistics
        self.dropout = dropout
        out_channels = in_channels
        self.a_input = False if a_channels is None else True
        if self.a_input:
            self.nin = WeightNormConv2d(a_channels, in_channels, filter_size=3, stride=1, padding='SAME')
            in_channels= in_channels + out_channels

        self.conv = WeightNormConv2d(in_channels, out_channels, filter_size=3, stride=1, padding='SAME')
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x, a=None):
        residual = x
        if a is not None and self.a_input:
            a = self.nin(F.relu(a))
            residual = torch.cat([residual, a], dim=1)

        residual = F.relu(residual)
        residual = F.dropout(residual, p=self.dropout, training=self.training)
        residual=self.conv(residual)
        residual = self.bn(residual)

        return x + residual

def adaptive_max_pool2d(x, kernel_size=2, stride=2):
    n, c, h, w = x.size()
    padding_h = 1 if h % 2 != 0 else 0
    padding_w = 1 if w % 2 != 0 else 0
    return F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=(padding_h, padding_w))

class post_enc(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, n_residual_blocks=2, num_scales=7, init_filters=64, conv=WeightNormConv2d):
        super().__init__()
        self.n_residual_blocks=n_residual_blocks
        self.num_scales=num_scales
        self.nin=WeightNormConv2d(in_channels_1+in_channels_2, init_filters, filter_size=3, stride=1, padding='SAME')
        self.res_blocks = nn.ModuleList(
    [ResidualBlock(init_filters) for _ in range(2)] +
    [ResidualBlock(init_filters*2) for _ in range(2)] +
    [ResidualBlock(init_filters*4) for _ in range((self.num_scales-2) * 2)]
)
        self.downsample_convs = nn.ModuleList(
            [conv(init_filters, init_filters*2)] + 
            [conv(init_filters*2, init_filters*4)] + 
            [conv(init_filters*4, init_filters*4) for _ in range(num_scales-1-2)]
)
    
    def forward(self, x, c):
        hs=[]
        xc=torch.cat((x, c), dim=1)
        h=self.nin(xc)
        for l in range(self.num_scales):
            for i in range(self.n_residual_blocks):
                res = self.res_blocks[self.n_residual_blocks*l+i]
                h=res(h)
                hs.append(h)
            if l+1 < self.num_scales:
                conv = self.downsample_convs[l]
                h = conv(h)
                h = F.relu(h)
                h = adaptive_max_pool2d(h, kernel_size=2, stride=2)
        return hs


class prior_enc(nn.Module):
    def __init__(self, in_channels, n_residual_blocks=2, num_scales=7, init_filters=64, conv=WeightNormConv2d):
        super().__init__()
        self.n_residual_blocks = n_residual_blocks
        self.num_scales = num_scales
        self.nin = WeightNormConv2d(in_channels, init_filters, filter_size=3, stride=1, padding='SAME')
        self.res_blocks = nn.ModuleList(
    [ResidualBlock(init_filters) for _ in range(2)] +
    [ResidualBlock(init_filters*2) for _ in range(2)] +
    [ResidualBlock(init_filters*4) for _ in range((self.num_scales-2) * 2)]
)
        self.downsample_convs = nn.ModuleList(
            [conv(init_filters, init_filters*2)] + 
            [conv(init_filters*2, init_filters*4)] + 
            [conv(init_filters*4, init_filters*4) for _ in range(num_scales-1-2)]
)

    def forward(self, x):
        hs=[]
        h=self.nin(x)
        for l in range(self.num_scales):
            for i in range(self.n_residual_blocks):
                res = self.res_blocks[self.n_residual_blocks*l+i]
                h=res(h)
                hs.append(h)
            if l+1 < self.num_scales:
                conv = self.downsample_convs[l]
                h = conv(h)
                h = F.relu(h)
                h = adaptive_max_pool2d(h, kernel_size=2, stride=2)
        return hs


class dec(nn.Module):
    def __init__(self, n_residual_blocks=2, num_scales=7, init_filters=256, sample1_channels=1, sample2_channels=2, conv=WeightNormConv2d):
        super().__init__()
        self.num_scales = num_scales
        self.n_residual_blocks = n_residual_blocks
        self.nin = WeightNormConv2d(init_filters, init_filters, filter_size=3, stride=1, padding='SAME')
        self.merge_conv_blocks = nn.ModuleList([conv(init_filters + sample1_channels, init_filters), conv(init_filters + sample2_channels, init_filters)])
        #这里的conv层用于融合特征图和抽样（或分布均值，直接融合分布均值，则此时为决定性网络）
        self.upsample_conv_blocks = nn.ModuleList(
            [conv(init_filters, init_filters) for _ in range(num_scales-1-2)] +
            [conv(init_filters, init_filters//2)] +
            [conv(init_filters//2, init_filters//4)]
    )        #上采样过程中压缩channel数的conv层，与encode过程正好相反，形成典型的U-Net架构
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(init_filters, init_filters) for _ in range((self.num_scales-2) * 2)] + 
            [ResidualBlock(init_filters//2, init_filters//2) for _ in range(2)] + 
            [ResidualBlock(init_filters//4, init_filters//4) for _ in range(2)]
    )
    
    def forward(self, enc_feature, latent_z, latent_mean, sample=[True,True,True,True,True,True,True]):
        dec_temp = self.nin(enc_feature[-1])
        for l in range(self.num_scales):
            for _ in range(self.n_residual_blocks//2):
                res_block = self.res_blocks[l*self.n_residual_blocks]
                dec_temp = res_block(dec_temp, enc_feature.pop())
            if l<2:
                if sample[l] == False:
                    z_prior = latent_mean[l]
                if sample[l] == True:
                    z_prior = latent_z[l]
                dec_temp = torch.cat((dec_temp, z_prior), dim=1)
                merge_conv_block = self.merge_conv_blocks[l]
                dec_temp = merge_conv_block(dec_temp)
            else:
                pass
            res_block = self.res_blocks[l*self.n_residual_blocks + 1]
            dec_temp = res_block(dec_temp, enc_feature.pop())
            if l+1 < self.num_scales:
                upsample_conv_block = self.upsample_conv_blocks[l]
                dec_temp = upsample_conv_block(dec_temp)
                x_up = enc_feature[-1]
                dec_temp = F.interpolate(dec_temp, size=x_up.shape[2:], mode='bilinear', align_corners=True)
        return dec_temp
    

class latent_encoder(nn.Module):
    def __init__(self, num_scale=7, init_filters=256, latent_scales_dims=[1,2,0,0,0,0,0], conv=WeightNormConv2d):
        super().__init__()
        self.num_scale = num_scale
        self.latent_parm_blocks = nn.ModuleList([conv(init_filters, latent_scales_dims[0]), conv(init_filters, latent_scales_dims[1])])
    def forward(self, features):
        latent_mv = []
        latent_z = []
        for l in range(self.num_scale):
            if l<2:
                latent_parm_block = self.latent_parm_blocks[l]
                mean_vae = latent_parm_block(features[-(2*l+1)])
                latent_mv.append(mean_vae)
                eps = torch.randn_like(mean_vae)
                z = mean_vae + eps
                latent_z.append(z)
        return latent_mv, latent_z


class ConvL2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvL2, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2)
        nn.init.trunc_normal_(self.conv.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.conv.bias)
        self.weight = self.conv.weight
    def forward(self, x):
        return self.conv(x)


class FClayerL2(nn.Module):
    def __init__(self, in_features, out_features):
        super(FClayerL2, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        nn.init.trunc_normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc.bias)
        self.weight = self.fc.weight
    def forward(self, x):
        return self.fc(x)

def _apply_pool(self, tensor, kernel_size=3, stride=2):
    # 假设我们要在右侧填充 1 列，在下方填充 1 行
    padding = (0, 1, 0, 1)  # (左, 右, 上, 下)
    tensor_padded = F.pad(tensor, padding, mode='constant', value=0)
    # 应用无填充的池化操作
    tensor_pooled = F.max_pool2d(tensor_padded, kernel_size=kernel_size, stride=stride, padding=0)
    return tensor_pooled

class ResidualBlock_v2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        half_num_out = max(int(out_channels/2), 1)
        self.conv_lists = nn.ModuleList([ConvL2(in_channels, half_num_out, kernel_size=1, stride=1),
                                 ConvL2(half_num_out, half_num_out, kernel_size=3, stride=1),
                                 ConvL2(half_num_out, out_channels, kernel_size=1, stride=1)])
        if in_channels != out_channels:
            self.conv_lists.append(ConvL2(in_channels, out_channels, kernel_size=1, stride=1))
        self.bn_lists = nn.ModuleList([nn.BatchNorm2d(in_channels)] +
                                      [nn.BatchNorm2d(half_num_out) for _ in range(2)])
    
    def forward(self, x):
        c = x
        for l in range(3):
            bn = self.bn_lists[l]
            c = bn(c)
            c = F.relu(c)
            conv = self.conv_lists[l]
            c = conv(c)
        if self.in_channels != self.out_channels:
            conv = self.conv_lists[-1]
            s = conv(s)
        else:
            s = x
        x = c + s
        return x


class build_after(nn.Module):
    def __init__(self, in_channels, _builder_num_residual_blocks=1):
        super(build_after, self).__init__()
        self._builder_num_residual_blocks = _builder_num_residual_blocks
        self.res_blocks = nn.ModuleList([ResidualBlock_v2(in_channels, in_channels) for _ in range(self._builder_num_residual_blocks)])
        self.conv_lists = nn.ModuleList([ConvL2(in_channels, in_channels, kernel_size=1, stride=1),
                                         ConvL2(in_channels, 2, kernel_size=1, stride=1)])
        self.bn = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        for l in range(self._builder_num_residual_blocks):
            res_block = self.res_blocks[l]
            x = res_block(x)
        conv1 = self.conv_lists[0]
        x = conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        conv2 = self.conv_lists[1]
        gmap = conv2(x)
        n, c, h, w = gmap.size()
        gmap = gmap.view(n, -1)
        gmap = F.softmax(gmap, dim=1)
        gmap = gmap.view(n, c, h, w)
        return gmap



#********************************************************************DenseNet部分***************************************************************************#

class DenseLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, growth_channels, kernels_size=[1, 3],
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, conv_layer=ConvL2):

        super().__init__()

        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            act_layer(),
            conv_layer(in_channels, hidden_channels, kernels_size[0])
        )
        self.conv2 = nn.Sequential(
            norm_layer(hidden_channels),
            act_layer(),
            conv_layer(hidden_channels, growth_channels, kernels_size[1])
        )

    def forward(self, x):
        x_ = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat((x, x_), dim=1)
        return x


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_channels,
                 dense_layer=DenseLayer, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, conv_layer=ConvL2):

        super().__init__()

        self.in_channels = in_channels
        self.growth_channels = growth_channels
        self.channels_ls = self.get_channels_ls(in_channels, growth_channels, num_layers)
        self.layers = nn.ModuleList(
            [dense_layer(
                in_channels, hidden_channels, growth_channels,
                norm_layer=norm_layer, act_layer=act_layer, conv_layer=conv_layer
            ) for in_channels, hidden_channels in zip(*self.channels_ls)]
        )

    def get_channels_ls(self, in_channels, growth_channels, num_layers):
        in_channels_ls = []
        hidden_channels_ls = []
        for _ in range(num_layers):
            in_channels_ls.append(in_channels)

            hidden_channels = min(in_channels, 4 * growth_channels)
            hidden_channels_ls.append(hidden_channels)

            in_channels += growth_channels
        return in_channels_ls, hidden_channels_ls

    @property
    def out_channels(self):
        in_channels_ls, _ = self.channels_ls
        return in_channels_ls[-1] + self.growth_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class densenet(nn.Module):
    def __init__(self, in_channels, growth_channels=8, compress_ratio=0.5, dense_block_sizes=[4, 4, 4, 4],
                 dense_block=DenseBlock, dense_layer=DenseLayer, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, conv_layer=ConvL2, pool_layer=nn.AvgPool2d):
        super().__init__()

        self.in_channels = in_channels
        self.growth_channels = growth_channels

        # build dense blocks
        dense_blk_ls = []
        for blk_size in dense_block_sizes:
            dense_blk_ls.append(
                dense_block(blk_size, in_channels, growth_channels,
                            dense_layer=dense_layer, norm_layer=norm_layer, act_layer=act_layer, conv_layer=conv_layer)
            )
            in_channels = int(dense_blk_ls[-1].out_channels * compress_ratio)
        self.dense_blocks = nn.ModuleList(dense_blk_ls)

        # build pooling blocks
        pooling_blk_ls = []
        for i in range(len(dense_blk_ls) - 1):
            in_channels = dense_blk_ls[i].out_channels
            pooling_blk = nn.Sequential(
                norm_layer(in_channels),
                act_layer(),
                conv_layer(in_channels, int(in_channels * compress_ratio), kernel_size=1),
                pool_layer(2, 2)
            )
            pooling_blk_ls.append(pooling_blk)
        self.pooing_blocks = nn.ModuleList(pooling_blk_ls)

        # build global average block
        self.global_avg = nn.Sequential(
            norm_layer(dense_blk_ls[-1].out_channels),
            act_layer(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            FClayerL2(dense_blk_ls[-1].out_channels, 2)
        )

    def forward(self, x):
        for i in range(len(self.dense_blocks) - 1):
            x = self.dense_blocks[i](x)
            x = self.pooing_blocks[i](x)
        x = self.dense_blocks[-1](x)
        x = self.global_avg(x)
        return x