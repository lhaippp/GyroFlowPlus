"""Defines the neural network, losss function and metrics"""
import json
import torch
import collections

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict
from abc import abstractmethod

from utils import flow_utils
from .nn_upsample import NeuralUpsampler, upsample2d_flow_as, FlowMaskEstimator


def conv(inp, out, k=3, s=1, d=1, isReLU=True):
    if isReLU:
        ret = nn.Sequential(nn.Conv2d(inp, out, k, s, padding=((k - 1) * d) // 2, dilation=d, bias=True), nn.LeakyReLU(0.1, inplace=True))
    else:
        ret = nn.Sequential(nn.Conv2d(inp, out, k, s, padding=((k - 1) * d) // 2, dilation=d, bias=True))
    return ret


class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()
        self.convs = nn.Sequential(conv(ch_in, 128, 3, 1, 1), conv(128, 128, 3, 1, 2), conv(128, 128, 3, 1, 4), conv(128, 96, 3, 1, 8),
                                   conv(96, 64, 3, 1, 16), conv(64, 32, 3, 1, 1), conv(32, 2, isReLU=False))

    def forward(self, x):
        return self.convs(x)


class FlowEstimator(nn.Module):
    def __init__(self, ch_in):
        super(FlowEstimator, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(128, 128)
        self.conv3 = conv(128 + 128, 96)
        self.conv4 = conv(96 + 128, 64)
        self.conv5 = conv(96 + 64, 32)
        # channels of the second last layer
        self.feat_dim = 32
        self.predict_flow = conv(64 + 32, 2, isReLU=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x3, x4], dim=1))
        flow = self.predict_flow(torch.cat([x4, x5], dim=1))
        return x5, flow


class FlowEstimator_subSpace(nn.Module):
    def __init__(self, ch_in):
        super(FlowEstimator_subSpace, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(128, 128)
        self.conv3 = conv(128 + 128, 96)
        self.conv4 = conv(96 + 128, 64)
        self.conv5 = conv(96 + 64, 32)
        # channels of the second last layer
        self.feat_dim = 32
        self.predict_flow = conv(64 + 32, 2, isReLU=False)

        self.sub_conv = nn.Conv2d(64 + 32, (64 + 32) // 2, 1)

    def subspace_project(self, x, vectors):
        b_, c_, h_, w_ = x.shape
        basis_vector_num = vectors.shape[1]

        V_t = vectors.view(b_, basis_vector_num, h_ * w_)
        V_t = V_t / (1e-6 + V_t.abs().sum(dim=2, keepdim=True))
        V = V_t.permute(0, 2, 1)

        mat = torch.bmm(V_t, V)
        mat_inv = torch.inverse(mat)
        project_mat = torch.bmm(mat_inv, V_t)
        input_ = x.view(b_, c_, h_ * w_)
        project_feature = torch.bmm(project_mat, input_.permute(0, 2, 1))
        output = torch.bmm(V, project_feature).permute(0, 2, 1).view(b_, c_, h_, w_)
        return output

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x3, x4], dim=1))

        # SSA
        sub = self.sub_conv(torch.cat([x4, x5], dim=1))
        x6 = self.subspace_project(torch.cat([x4, x5], dim=1), sub)

        # flow = self.predict_flow(torch.cat([x4, x5], dim=1))
        flow = self.predict_flow(x6)
        return x5, flow


class CostVolume(nn.Module):
    def __init__(self, d=4, *args, **kwargs):
        super(CostVolume, self).__init__()
        self.d = d
        self.out_dim = 2 * self.d + 1
        self.pad_size = self.d

    def forward(self, x1, x2):
        B, C, H, W = x1.size()

        x2 = F.pad(x2, [self.pad_size] * 4)
        cv = []
        for i in range(self.out_dim):
            for j in range(self.out_dim):
                cost = x1 * x2[:, :, i:(i + H), j:(j + W)]
                cost = torch.mean(cost, 1, keepdim=True)
                cv.append(cost)
        return torch.cat(cv, 1)


class CorrTorch_lkm(nn.Module):
    '''
    自己实现的correlation
    '''
    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1):
        assert pad_size == max_displacement
        assert stride1 == stride2 == 1
        super().__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.stride1 = stride1
        self.stride2 = stride2
        self.max_hdisp = max_displacement
        self.padlayer = nn.ConstantPad2d(pad_size, 0)

    def forward(self, in1, in2):
        bz, cn, hei, wid = in1.shape
        f1 = F.unfold(in1, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=self.stride1)
        f2 = F.unfold(in2, kernel_size=self.kernel_size, padding=self.kernel_size // 2,
                      stride=self.stride2)  # 在这一步抽取完了kernel以后做warping插值岂不美哉？
        matching_kernel_size = f2.shape[1]
        f2_ = torch.reshape(f2, (bz, matching_kernel_size, hei, wid))
        f2_ = torch.reshape(f2_, (bz * matching_kernel_size, hei, wid)).unsqueeze(1)

        searching_kernel_size = self.max_hdisp * 2 + 1
        f2 = F.unfold(f2_, kernel_size=searching_kernel_size, padding=searching_kernel_size // 2, stride=self.stride2)
        # tensor_tools.check_tensor(f2, 'f2_reunfold')
        _, search_window_size, window_number = f2.shape
        f2_ = torch.reshape(f2, (bz, matching_kernel_size, search_window_size, window_number))
        f2_2 = torch.transpose(f2_, dim0=1, dim1=2)
        window_number = search_window_size

        f1_2 = f1.unsqueeze(1)
        res = f2_2 * f1_2
        res = torch.mean(res, dim=2)
        res = torch.reshape(res, (bz, window_number, hei, wid))
        return res


class FeaturePyramidExtractor(nn.Module):
    def __init__(self, pyr_chans):
        super(FeaturePyramidExtractor, self).__init__()
        self.pyr_chans = pyr_chans
        self.convs = nn.ModuleList()
        for l, (ch_in, ch_out) in enumerate(zip(pyr_chans[:-1], pyr_chans[1:])):
            layer = nn.Sequential(conv(ch_in, ch_out, s=2), conv(ch_out, ch_out))
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)
        return feature_pyramid[::-1]


class PWCLite(nn.Module):
    def __init__(self, params):
        super(PWCLite, self).__init__()
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.upsample = params.upsample
        self.with_bk = True
        # self.fuse_layer = params.fuse_layer

        self.pyr_chans = [3, 16, 32, 64, 96, 128, 192]
        self.feature_pyramid_extractor = FeaturePyramidExtractor(self.pyr_chans)
        # correlation range
        self.d = 4
        self.output_level = 4
        # cost volume
        self.cost_volume = CostVolume(d=self.d)
        # self.cost_volume = CorrTorch_lkm()
        self.cv_dim = (self.d * 2 + 1)**2
        # neural upsampler
        self.upsampler = NeuralUpsampler()
        # 32 denotes the
        self.ch_inp = 32 + self.cv_dim + 2
        self.flow_estimator = FlowEstimator(self.ch_inp)
        self.context_net = ContextNetwork(self.flow_estimator.feat_dim + 2)

        self.conv_1x1 = nn.ModuleList([
            conv(192, 32, k=1, s=1, d=1),
            conv(128, 32, k=1, s=1, d=1),
            conv(96, 32, k=1, s=1, d=1),
            conv(64, 32, k=1, s=1, d=1),
            conv(32, 32, k=1, s=1, d=1)
        ])

        self.inter_results = {}
        self.with_gyro_field = True
        self.add_gyro_layer = params.add_gyro_layer
        self.homo_decoder_layer = params.homo_decoder_layer

    # fusion of gyro field and optical flow
    @abstractmethod
    def make_fuse_mask(self, x1, x2):
        pass

    def predict_flow(self, x1_pyrs, x2_pyrs, gyro_field=None):
        flow_pyrs = []

        batch_size, _, h_x1, w_x1 = x1_pyrs[0].size()
        dtype, device = x1_pyrs[0].dtype, x1_pyrs[0].device

        flow = torch.zeros(batch_size, 2, h_x1, w_x1, dtype=dtype, device=device)

        for l, (x1, x2) in enumerate(zip(x1_pyrs, x2_pyrs)):
            if self.add_gyro_layer == 0 and l == self.add_gyro_layer:
                flow = upsample2d_flow_as(gyro_field, flow, if_rate=True)
                x2_warp = flow_utils.flow_warp(x2, flow)
            else:
                flow = self.upsampler(flow, self.conv_1x1[l](x1), self.conv_1x1[l](x2))
                self.inter_results["flow_init_{}".format(l)] = torch.clone(flow)
                if l == self.add_gyro_layer and self.with_gyro_field and self.add_gyro_layer != 0:
                    gyro_field_rsz = upsample2d_flow_as(gyro_field, flow, if_rate=True)
                    # self.inter_results["flow_init_{}".format(l)] = flow
                    self.inter_results["gyro_field_{}".format(l)] = torch.clone(gyro_field_rsz)
                    flow += gyro_field_rsz
                    self.inter_results["flow_fuse_{}".format(l)] = torch.clone(flow)
                x2_warp = flow_utils.flow_warp(x2, flow)

            _cv = self.cost_volume(x1, x2_warp)
            _cv_relu = self.leakyRELU(_cv)

            x1 = self.conv_1x1[l](x1)
            _x_feat, flow_pred = self.flow_estimator(torch.cat([_cv_relu, x1, flow], dim=1))
            self.inter_results["flow_pred_{}".format(l)] = flow_pred
            flow += flow_pred

            flow_refine = self.context_net(torch.cat([_x_feat, flow], dim=1))
            self.inter_results["flow_refine_{}".format(l)] = flow_refine
            flow += flow_refine

            self.inter_results["flow_final_{}".format(l)] = flow
            flow_pyrs.append(flow)
            if l == self.output_level:
                break
        if self.upsample:
            flows = [F.interpolate(flow * 4, scale_factor=4, mode='bilinear', align_corners=True) for flow in flow_pyrs]
            self.inter_results["flows"] = flows
        return flows[::-1]

    def forward(self, data_batch, with_bk=True):
        x = data_batch['imgs']
        imgs = [x[:, 3 * i:3 * i + 3] for i in range(2)]
        x = [self.feature_pyramid_extractor(img) + [img] for img in imgs]

        gyro_field = data_batch["gyro_field"]
        # gyro_field = torch.zeros_like(gyro_field)

        # warp_img2 = flow_utils.flow_warp(imgs[1], gyro_field)
        # with imageio.get_writer("test.gif", mode='I', duration=0.5) as writer:
        #     x = np.concatenate((
        #         imgs[0][0].detach().cpu().numpy().transpose(1, 2, 0),
        #         imgs[0][0].detach().cpu().numpy().transpose(1, 2, 0),
        #     ), axis=1)
        #     y = np.concatenate((
        #         imgs[1][0].squeeze().cpu().detach().numpy().transpose(1, 2, 0),
        #         warp_img2[0].squeeze().cpu().detach().numpy().transpose(1, 2, 0),
        #     ), axis=1)
        #     writer.append_data(x)
        #     writer.append_data(y)

        # raise Exception("test!")

        res = {}
        res['flow_fw'] = self.predict_flow(x[0], x[1], gyro_field)
        if with_bk:
            res['flow_bw'] = self.predict_flow(x[1], x[0], -1 * gyro_field)

        res["inter_results"] = self.inter_results
        return res

    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


class UFlowSGF(PWCLite):
    def __init__(self, params):
        super(UFlowSGF, self).__init__(params=params)
        self.flow_predictor = FlowMaskEstimator(4, (8, 16, 32, 16, 8), 2)
        self.mask_predictor = FlowMaskEstimator(64, (32, 32, 32, 16, 8), 1)

        self.map_tau = params.map_tau

    def generate_fused_flow(self, x1, x2):
        input_feature = torch.cat((x1, x2), dim=1)
        flow = self.flow_predictor(input_feature)[1]
        assert flow.shape[1] == 2
        return flow

    def generate_map(self, x1, x2, layer=1):
        input_feature = torch.cat((x1, x2), dim=1)
        out = self.mask_predictor(input_feature)[1]

        interval_start = 0
        if layer == 4:
            interval_start = 0.75

        mask = interval_start + (1 - interval_start) * torch.sigmoid(out * self.map_tau)
        assert mask.shape[1] == 1
        return mask

    def self_guided_fusion_module(self, flow, gyro_field_rsz, x1, x2_warp, layer):
        fuse_flow = self.generate_fused_flow(flow, gyro_field_rsz)

        mask = self.generate_map(self.conv_1x1[layer](x1), self.conv_1x1[layer](x2_warp), layer)
        self.inter_results["map_fuse_{}".format(layer)] = torch.clone(mask)
        # import cv2
        # cv2.imwrite("inter_results/SGF_mask_{}.png".format(layer), mask[0].detach().cpu().numpy().squeeze() * 255)
        # print("layer {}, min: {} - mean: {} - max: {}".format(layer, torch.min(mask), torch.mean(mask), torch.max(mask)))

        flow = fuse_flow * mask + gyro_field_rsz * (1 - mask)
        return flow, mask

    def normalize_features(self, feature_list, normalize, center, moments_across_channels=True, moments_across_images=True):
        """Normalizes feature tensors (e.g., before computing the cost volume).
        Args:
          feature_list: list of torch tensors, each with dimensions [b, c, h, w]
          normalize: bool flag, divide features by their standard deviation
          center: bool flag, subtract feature mean
          moments_across_channels: bool flag, compute mean and std across channels, 看到UFlow默认是True
          moments_across_images: bool flag, compute mean and std across images, 看到UFlow默认是True

        Returns:
          list, normalized feature_list
        """

        # Compute feature statistics.
        statistics = collections.defaultdict(list)
        axes = [1, 2, 3] if moments_across_channels else [2, 3]  # [b, c, h, w]
        for feature_image in feature_list:
            mean = torch.mean(feature_image, dim=axes, keepdim=True)  # [b,1,1,1] or [b,c,1,1]
            variance = torch.var(feature_image, dim=axes, keepdim=True)  # [b,1,1,1] or [b,c,1,1]
            statistics['mean'].append(mean)
            statistics['var'].append(variance)

        if moments_across_images:
            # statistics['mean'] = ([tf.reduce_mean(input_tensor=statistics['mean'])] *
            #                       len(feature_list))
            # statistics['var'] = [tf.reduce_mean(input_tensor=statistics['var'])
            #                      ] * len(feature_list)
            statistics['mean'] = ([torch.mean(torch.stack(statistics['mean'], dim=0), dim=(0, ))] * len(feature_list))
            statistics['var'] = ([torch.var(torch.stack(statistics['var'], dim=0), dim=(0, ))] * len(feature_list))

        statistics['std'] = [torch.sqrt(v + 1e-16) for v in statistics['var']]

        # Center and normalize features.
        if center:
            feature_list = [f - mean for f, mean in zip(feature_list, statistics['mean'])]
        if normalize:
            feature_list = [f / std for f, std in zip(feature_list, statistics['std'])]
        return feature_list

    def predict_flow(self, x1_pyrs, x2_pyrs, gyro_field=None):
        flow_pyrs = []

        batch_size, _, h_x1, w_x1 = x1_pyrs[0].size()
        dtype, device = x1_pyrs[0].dtype, x1_pyrs[0].device

        flow = torch.zeros(batch_size, 2, h_x1, w_x1, dtype=dtype, device=device)

        for l, (x1, x2) in enumerate(zip(x1_pyrs, x2_pyrs)):
            if l == 0:
                x2_warp = x2
            else:
                flow = self.upsampler(flow, self.conv_1x1[l](x1), self.conv_1x1[l](x2))
                # self.inter_results["flow_init_{}".format(l)] = torch.clone(flow)

                gyro_field_rsz = upsample2d_flow_as(gyro_field, flow, if_rate=True)
                x2_warp = flow_utils.flow_warp(x2, gyro_field_rsz)
                # self.inter_results["gyro_field_{}".format(l)] = torch.clone(gyro_field_rsz)

                flow = self.self_guided_fusion_module(flow, gyro_field_rsz, x1, x2_warp, l)

                # self.inter_results["flow_fuse_{}".format(l)] = torch.clone(flow)
                x2_warp = flow_utils.flow_warp(x2, flow)

            # cost volume normalized
            x1_normalized, x2_warp_normalized = self.normalize_features([x1, x2_warp],
                                                                        normalize=True,
                                                                        center=True,
                                                                        moments_across_channels=False,
                                                                        moments_across_images=False)

            _cv = self.cost_volume(x1_normalized, x2_warp_normalized)
            _cv_relu = self.leakyRELU(_cv)

            x1 = self.conv_1x1[l](x1)
            _x_feat, flow_pred = self.flow_estimator(torch.cat([_cv_relu, x1, flow], dim=1))
            # self.inter_results["flow_pred_{}".format(l)] = flow_pred
            flow += flow_pred

            flow_refine = self.context_net(torch.cat([_x_feat, flow], dim=1))
            # self.inter_results["flow_refine_{}".format(l)] = flow_refine
            flow += flow_refine

            # self.inter_results["flow_final_{}".format(l)] = flow
            flow_pyrs.append(flow)
            if l == self.output_level:
                break
        if self.upsample:
            flows = [F.interpolate(flow * 4, scale_factor=4, mode='bilinear', align_corners=True) for flow in flow_pyrs]
            # self.inter_results["flows"] = flows
        return flows[::-1]


class UFlow(UFlowSGF):
    def self_guided_fusion_module(self, flow, gyro_field_rsz, x1, x2_warp, layer):
        return flow


class UFlowSGFFuse(UFlowSGF):
    def self_guided_fusion_module(self, flow, gyro_field_rsz, x1, x2_warp, layer):
        fuse_flow = self.generate_fused_flow(flow, gyro_field_rsz)
        return fuse_flow


class UFlowSGFMap(UFlowSGF):
    def self_guided_fusion_module(self, flow, gyro_field_rsz, x1, x2_warp, layer):
        mask = self.generate_map(self.conv_1x1[layer](x1), self.conv_1x1[layer](x2_warp))
        flow = flow * mask + gyro_field_rsz * (1 - mask)
        return flow


class UFlowSGFGyroHomo(UFlowSGF):
    def __init__(self, params):
        super(UFlowSGFGyroHomo, self).__init__(params=params)
        self.gyro_homo_fuser = FlowMaskEstimator(4, (8, 16, 32, 16, 8), 2)
        self.refine_map_generator = FlowMaskEstimator(64, (32, 32, 32, 16, 8), 1)

        self.basis = flow_utils.gen_basis(*params.data_aug.para_crop)  # 8,2,h,w --> 1, 8, 2*h*w
        print("self.basis shape is: ", self.basis.shape)
        self.sp_layer = Subspace(64)
        self.weight_generator = nn.Conv2d(64, 8, kernel_size=1, stride=1, padding=0, groups=8, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.inter_results["sgf_fusion_map"] = []

        self.refine_mask_weight = params.refine_mask_weight
        self.detach_homo_field = params.detach_homo_field
        self.detach_sgf_map = params.detach_sgf_map

    def gyro_homo_module(self, gyro_field, x1, x2, mask_prev):
        x2_warp = flow_utils.flow_warp(x2, gyro_field)

        # predict a mask to outline foregrounds
        mask_sgf = torch.detach(mask_prev)
        # self.inter_results["sgf_fusion_map"].append(mask)

        refine_mask = self.refine_map_generator(torch.cat([x1 * mask_sgf, x2 * mask_sgf], 1))[-1]
        refine_mask = torch.sigmoid(refine_mask)
        refine_mask = refine_mask - 0.5
        mask = mask_sgf + self.refine_mask_weight * refine_mask
        mask = torch.clamp(mask, 0, 1)

        # self.inter_results["map_homo"] = torch.clone(self.refine_mask_weight * refine_mask)
        # print("res_map_homo: min {} - mean {} - max {}".format(torch.min(self.inter_results["map_homo"]),
        #                                                        torch.mean(self.inter_results["map_homo"]),
        #                                                        torch.max(self.inter_results["map_homo"])))
        # print("map_homo: min {} - mean {} - max {}".format(torch.min(mask), torch.mean(mask), torch.max(mask)))

        x1_mask_fg = x1 * mask
        x2_warp_mask_fg = x2_warp * mask

        # pre-process weight flow
        n, _, h, w = gyro_field.shape
        _, _, _h, _w = self.basis.shape

        if h != _h or w != _w:
            basis_flow = upsample2d_flow_as(self.basis, gyro_field, if_rate=True)
        basis_flow = basis_flow.unsqueeze(0).reshape(1, 8, -1)

        x = self.sp_layer(torch.cat([x1_mask_fg, x2_warp_mask_fg], 1))
        weights = self.weight_generator(x)
        weight_f = self.pool(weights).squeeze(3)

        H_flow_f = (basis_flow.to(weight_f.device) * weight_f).sum(1).reshape(n, 2, h, w)

        gyro_homo = self.gyro_homo_fuser(torch.cat([H_flow_f, gyro_field], 1))[-1]
        return gyro_homo, mask

    def predict_flow(self, x1_pyrs, x2_pyrs, gyro_field=None):
        homo_pyrs, flow_pyrs = [], []

        batch_size, _, h_x1, w_x1 = x1_pyrs[0].size()
        dtype, device = x1_pyrs[0].dtype, x1_pyrs[0].device

        flow = torch.zeros(batch_size, 2, h_x1, w_x1, dtype=dtype, device=device)
        mask, mask_prev = None, None

        gyro_homo = upsample2d_flow_as(gyro_field, flow, if_rate=True)

        for l, (x1, x2) in enumerate(zip(x1_pyrs, x2_pyrs)):
            if l == 0:
                x2_warp = x2
            else:
                flow = self.upsampler(flow, self.conv_1x1[l](x1), self.conv_1x1[l](x2))
                self.inter_results["flow_init_{}".format(l)] = torch.clone(flow)

                if gyro_homo.shape[-1] != flow.shape[-1]:
                    gyro_homo = upsample2d_flow_as(gyro_field, flow, if_rate=True)

                if l == self.homo_decoder_layer:
                    _, _, h, w = gyro_homo.size()
                    # mask represents the region where flow-net should be concentrated
                    # as opposite, 1 - mask represents the region where gyro-net should be concentraeted
                    mask = 1 - F.interpolate(mask_prev, [h, w], mode="nearest")

                    gyro_homo_homo_decoder, homo_mask = self.gyro_homo_module(gyro_homo, self.conv_1x1[l](x1), self.conv_1x1[l](x2), mask)
                    # gyro_homo += gyro_homo_residual
                    homo_pyrs.append(gyro_homo_homo_decoder)

                    # gyro_homo = torch.detach(gyro_homo_homo_decoder) if self.detach_homo_field else gyro_homo

                x2_warp = flow_utils.flow_warp(x2, gyro_homo)
                # self.inter_results["gyro_field_{}".format(l)] = torch.clone(gyro_field_rsz)

                flow, mask_prev = self.self_guided_fusion_module(flow, gyro_homo, x1, x2_warp, l)

                self.inter_results["flow_fuse_{}".format(l)] = torch.clone(flow)
                x2_warp = flow_utils.flow_warp(x2, flow)

            # cost volume normalized
            x1_normalized, x2_warp_normalized = self.normalize_features([x1, x2_warp],
                                                                        normalize=True,
                                                                        center=True,
                                                                        moments_across_channels=False,
                                                                        moments_across_images=False)

            _cv = self.cost_volume(x1_normalized, x2_warp_normalized)
            _cv_relu = self.leakyRELU(_cv)

            x1 = self.conv_1x1[l](x1)
            _x_feat, flow_pred = self.flow_estimator(torch.cat([_cv_relu, x1, flow], dim=1))
            self.inter_results["flow_pred_{}".format(l)] = flow_pred
            flow += flow_pred

            flow_refine = self.context_net(torch.cat([_x_feat, flow], dim=1))
            # self.inter_results["flow_refine_{}".format(l)] = flow_refine
            flow += flow_refine

            self.inter_results["flow_final_{}".format(l)] = flow
            flow_pyrs.append(flow)
            if l == self.output_level:
                break
        if self.upsample:
            flows = [F.interpolate(flow * 4, scale_factor=4, mode='bilinear', align_corners=True) for flow in flow_pyrs]
            homos = [upsample2d_flow_as(homo, flows[-1], if_rate=True) for homo in homo_pyrs]
            # print("homo shape is ", homos[0].shape)
            # self.inter_results["flows"] = flows
        return flows[::-1], homos[::-1], homo_mask

    def forward(self, data_batch, with_bk=True):
        x = data_batch['imgs']
        imgs = [x[:, 3 * i:3 * i + 3] for i in range(2)]
        x = [self.feature_pyramid_extractor(img) + [img] for img in imgs]

        gyro_field = data_batch["gyro_field"]

        res = {}
        res['flow_fw'], res['homo_fw'], res['homo_mask_fw'] = self.predict_flow(x[0], x[1], gyro_field)
        if with_bk:
            res['flow_bw'], res['homo_bw'], res['homo_mask_bw'] = self.predict_flow(x[1], x[0], -1 * gyro_field)

        res["inter_results"] = self.inter_results
        return res


def subspace_project(input, vectors):
    b_, c_, h_, w_ = input.shape
    basis_vector_num = vectors.shape[1]
    V_t = vectors.view(b_, basis_vector_num, h_ * w_)
    V_t = V_t / (1e-6 + V_t.abs().sum(dim=2, keepdim=True))
    V = V_t.permute(0, 2, 1)
    mat = torch.bmm(V_t, V)
    mat_inv = torch.inverse(mat)
    project_mat = torch.bmm(mat_inv, V_t)
    input_ = input.view(b_, c_, h_ * w_)
    project_feature = torch.bmm(project_mat, input_.permute(0, 2, 1))
    output = torch.bmm(V, project_feature).permute(0, 2, 1).view(b_, c_, h_, w_)
    return output


class Subspace(nn.Module):
    def __init__(self, ch_in, k=16, use_SVD=True, use_PCA=False):
        super(Subspace, self).__init__()
        self.k = k
        self.Block = SubspaceBlock(ch_in, self.k)
        self.use_SVD = use_SVD
        self.use_PCA = use_PCA

    def forward(self, x):
        sub = self.Block(x)
        x = subspace_project(x, sub)
        return x


class SubspaceBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(SubspaceBlock, self).__init__()

        self.relu = nn.LeakyReLU(inplace=True)

        self.conv0 = conv(inplanes, planes, k=1, s=1, d=1, isReLU=False)
        self.bn0 = nn.BatchNorm2d(planes)
        self.conv1 = conv(planes, planes, k=1, s=1, d=1, isReLU=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv(planes, planes, k=1, s=1, d=1, isReLU=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = self.relu(self.bn0(self.conv0(x)))

        out = self.conv1(residual)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        return out


if __name__ == "__main__":
    json_path = "/data/fusion_gyro_optical_flow/deep_gyro_of_fusion/work/SGF_ablation_study/experiments/experiment_SGF_ablation_study_GOF_clean_final_400_case/exp_0/params.json"
    with open(json_path) as f:
        params = EasyDict(json.load(f))

    net = UFlowSGF(params)

    data_batch = {}
    data_batch["imgs"] = torch.from_numpy(np.random.randn(1, 6, 768, 1024).astype(np.float32))
    data_batch["gyro_field"] = torch.from_numpy(np.random.randn(1, 2, 768, 1024).astype(np.float32))

    res = net(data_batch)

    flow_fw = res["flow_fw"]

    [print(i.shape) for i in flow_fw]
