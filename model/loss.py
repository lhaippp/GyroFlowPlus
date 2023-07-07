import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import flow_utils


class LossL1(nn.Module):
    def __init__(self):
        super(LossL1, self).__init__()
        self.loss = nn.L1Loss()

    def __call__(self, input, target):
        return self.loss(input, target)


class LossL2(nn.Module):
    def __init__(self):
        super(LossL2, self).__init__()
        self.loss = nn.MSELoss()

    def __call__(self, input, target):
        return self.loss(input, target)


class LossSmoothL1(nn.Module):
    def __init__(self):
        super(LossSmoothL1, self).__init__()
        self.loss = nn.SmoothL1Loss()

    def __call__(self, input, target):
        return self.loss(input, target)


class LossCrossEntropy(nn.Module):
    def __init__(self, weight=None):
        super(LossCrossEntropy, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss(weight=weight)

    def __call__(self, input, target, weight=None):
        return self.loss(input, target)


def SSIM(x, y, md=1):
    patch_size = 2 * md + 1
    C1 = 0.01**2
    C2 = 0.03**2

    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist


def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def smooth_grad_1st(flo, image, alpha):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)

    loss_x = weights_x * dx.abs() / 2.0
    loss_y = weights_y * dy.abs() / 2
    return loss_x.mean() / 2.0 + loss_y.mean() / 2.0


def TernaryLoss(im, im_warp, max_distance=1):
    patch_size = 2 * max_distance + 1

    def _rgb_to_grayscale(image):
        grayscale = image[:, 0, :, :] * 0.2989 + image[:, 1, :, :] * 0.5870 + image[:, 2, :, :] * 0.1140
        return grayscale.unsqueeze(1)

    def _ternary_transform(image):
        intensities = _rgb_to_grayscale(image) * 255
        out_channels = patch_size * patch_size
        w = torch.eye(out_channels).view((out_channels, 1, patch_size, patch_size))
        weights = w.type_as(im)
        patches = F.conv2d(intensities, weights, padding=max_distance)
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
        return transf_norm

    def _hamming_distance(t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist)
        dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum
        return dist_mean

    def _valid_mask(t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    t1 = _ternary_transform(im)
    t2 = _ternary_transform(im_warp)
    dist = _hamming_distance(t1, t2)
    mask = _valid_mask(im, max_distance)
    return dist * mask


class UnFlowLoss(nn.modules.Module):
    def __init__(self, params):
        super(UnFlowLoss, self).__init__()
        self.params = params

    def photo_loss(self, img1, img2_warp, occu_mask1):
        l1_loss = self.params.loss.l1 * (img1 - img2_warp).abs() * occu_mask1
        # print("l1_loss: ", l1_loss.mean())
        ssim_loss = self.params.loss.ssim * SSIM(img1 * occu_mask1, img2_warp * occu_mask1)
        # print("ssim_loss: ", ssim_loss.mean())
        tenary_loss = self.params.loss.tenary * TernaryLoss(img1 * occu_mask1, img2_warp * occu_mask1)
        # print("tenary_loss: ", tenary_loss.mean())
        return sum([l1_loss.mean(), ssim_loss.mean(), tenary_loss.mean()])

    def smooth_loss(self, flow, img):
        loss = smooth_grad_1st(flow, img, 10)
        return sum([loss.mean()])

    def forward(self, output, target, epoch=0):
        flows_fw, flows_bw = output["flow_fw"], output["flow_bw"]

        # flows_fw = [torch.ones_like(i) for i in flows_fw]
        # flows_bw = [torch.ones_like(i) for i in flows_bw]

        flow_pyrs = [torch.cat([flow_fw, flow_bk], 1) for flow_fw, flow_bk in zip(flows_fw, flows_bw)]
        img1, img2 = target[:, :3], target[:, 3:]

        self.pyramid_occu_mask1 = []
        self.pyramid_occu_mask2 = []

        if self.params.occ_from_back:
            occu_mask1 = 1 - flow_utils.get_occu_mask_backward(flow_pyrs[0][:, 2:], th=0.2)
            occu_mask2 = 1 - flow_utils.get_occu_mask_backward(flow_pyrs[0][:, :2], th=0.2)
        else:
            occu_mask1 = 1 - flow_utils.get_occu_mask_bidirection(flow_pyrs[0][:, :2], flow_pyrs[0][:, 2:])
            occu_mask2 = 1 - flow_utils.get_occu_mask_bidirection(flow_pyrs[0][:, 2:], flow_pyrs[0][:, :2])

        pyramid_smooth_losses = []
        pyramid_warp_losses = []

        for i, flow in enumerate(flow_pyrs):
            b, c, h, w = flow.size()
            if i == 0:
                s = min(h, w)
            if i == 4:
                pyramid_smooth_losses.append(0)
                pyramid_warp_losses.append(0)
                continue

            img1_rsz = F.interpolate(img1, (h, w), mode="area")
            img2_rsz = F.interpolate(img2, (h, w), mode="area")

            img1_warp = flow_utils.flow_warp(img2_rsz, flow[:, :2], pad="border")
            img2_warp = flow_utils.flow_warp(img1_rsz, flow[:, 2:], pad="border")

            if i != 0:
                occu_mask1 = F.interpolate(occu_mask1, (h, w), mode="nearest")
                occu_mask2 = F.interpolate(occu_mask2, (h, w), mode="nearest")

            self.pyramid_occu_mask1.append(occu_mask1)
            self.pyramid_occu_mask2.append(occu_mask2)

            if epoch < 50:
                # print("Forbideen occlusion mask")
                occu_mask1 = occu_mask2 = torch.tensor((), dtype=occu_mask2.dtype, device=occu_mask2.device).new_ones((occu_mask2.size()))

            photo_loss = self.photo_loss(img1_rsz, img1_warp, occu_mask1)
            smooth_loss = self.smooth_loss(flow[:, :2] / s, img1_rsz)

            # backward warping
            photo_loss += self.photo_loss(img2_rsz, img2_warp, occu_mask2)
            smooth_loss += self.smooth_loss(flow[:, 2:] / s, img2_rsz)

            photo_loss /= 2
            smooth_loss /= 2

            pyramid_smooth_losses.append(photo_loss)
            pyramid_warp_losses.append(smooth_loss)

            del photo_loss
            del smooth_loss

        _photo_loss = sum(pyramid_smooth_losses)
        _smooth_loss = 50 * pyramid_warp_losses[0]
        return _photo_loss + _smooth_loss


def fetch_loss(params):
    if params.loss_type == "UnFlowLoss":
        unFlowLoss = UnFlowLoss(params)
        return unFlowLoss


def compute_losses(data, endpoints, manager):
    losses = {}

    if manager.params.loss_type == "UnFlowLoss":
        # compute losses
        unFlowLoss = manager.unFlowLoss
        B = data["imgs"].size()[0]
        losses['total'] = unFlowLoss(endpoints, data["imgs"], manager.train_status["epoch"])
    elif manager.params.loss_type == "UnFlowLoss_HomoLoss":
        # compute losses
        unFlowLoss = manager.unFlowLoss
        homoLoss = manager.homoLoss
        B = data["imgs"].size()[0]
        losses['flow'] = unFlowLoss(endpoints, data["imgs"], manager.train_status["epoch"])
        losses['homo'] = homoLoss(endpoints, data["imgs"])
        losses['total'] = losses['homo'] + losses['flow']
    else:
        raise NotImplementedError

    for k, v in losses.items():
        manager.loss_status[k].update(val=v.item(), num=B)
        manager.train_status['print_str'] += '%s: %.4f(%.4f)' % (k, manager.loss_status[k].val, manager.loss_status[k].avg)
    return losses


def get_grid(batch_size, H, W, start):
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    ones = torch.ones_like(xx)
    grid = torch.cat((xx, yy, ones), 1).float()
    if torch.cuda.is_available():
        grid = grid.cuda()
    grid[:, :2, :, :] = grid[:, :2, :, :] + start  # 加上patch在原图内的偏移量
    return grid


def geometricDistance(flow, data_img, data):
    img_indices = get_grid(batch_size=data_img.shape[0], H=data_img.shape[2], W=data_img.shape[3], start=0)
    # print(img_indices.shape)
    vgrid = img_indices[:, :2, ...]
    grid_warp = vgrid + flow

    errors = 0
    # points = 6

    for i, _ in enumerate(data):
        points_LR = data[i]

        x1, y1, x2, y2 = points_LR[0][0], points_LR[0][1], points_LR[1][0], points_LR[1][1]

        x1_proj = grid_warp[:, 0, int(y1), int(x1)].detach().cpu().numpy()
        y1_proj = grid_warp[:, 1, int(y1), int(x1)].detach().cpu().numpy()

        error = np.sqrt(np.square(x1_proj - x2.detach().cpu().numpy()) + np.square(y1_proj - y2.detach().cpu().numpy()))
        errors += error

    err_avg = errors / len(data)
    return err_avg


def upsample2d_flow_as(inputs, target_as, mode="bilinear", if_rate=False):
    _, _, h, w = target_as.size()
    res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    if if_rate:
        _, _, h_, w_ = inputs.size()
        u_scale = (w / w_)
        v_scale = (h / h_)
        u, v = res.chunk(2, dim=1)
        u = u * u_scale
        v = v * v_scale
        res = torch.cat([u, v], dim=1)
    return res


def compute_metrics(data, endpoints, manager, name="epe"):
    metrics = {}
    # compute metrics
    B = data["img1"].size()[0]
    flow_fw = endpoints["flow_fw"][0]

    gt_flow = data["gt_flow"]
    epe, pck1, pck5 = flow_utils.flow_error_avg(flow_fw, gt_flow)
    metrics[name] = epe
    metrics[name + '_pck1'] = pck1
    metrics[name + '_pck5'] = pck5

    for k, v in metrics.items():
        manager.val_status[k].update(val=v.item(), num=B)
    return metrics


def compute_test_metrics(data, endpoints, homo_eval=False):
    ret = []

    gt_flow = data["gt_flow"]
    flow_fw = endpoints["flow_fw"][0]
    buf = flow_utils.flow_error_avg(flow_fw, gt_flow)

    [ret.append(i) for i in buf]

    if homo_eval:
        _, _, h, _ = gt_flow.shape
        # mask = data["mask"]
        homo_gt = data["homo_field"]
        homo_fw = endpoints["homo_fw"][0]
        epe_homo, pck1_homo, pck5_homo = flow_utils.flow_error_avg(homo_fw[:, :, :h // 2], homo_gt[:, :, :h // 2])
        # epe_homo, _, _ = flow_utils.flow_error_avg(homo_fw, homo_gt)
        # epe_homo, _, _ = flow_utils.flow_error_avg(homo_fw * mask, gt_flow * mask)

        ret.append([epe_homo, pck1_homo, pck5_homo])
    return ret


def update_metrics(ret, metrics, B, manager, name, homo_eval=False):
    if homo_eval:
        metrics[name + '_homo'] = ret[-1][0]
        metrics[name + '_homo_pck1'] = ret[-1][1]
        metrics[name + '_homo_pck5'] = ret[-1][2]

    epe, pck1, pck5 = ret[0], ret[1], ret[2]
    metrics[name + '_epe'] = epe
    metrics[name + '_pck1'] = pck1
    metrics[name + '_pck5'] = pck5

    for k, v in metrics.items():
        manager.test_status[k].update(val=v.item(), num=B)
