from collections import defaultdict
import numpy as np
import os
from PIL import Image
import argparse
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import cdist
from medpy.metric.binary import hd95

# def hd95(pred, true, distance="euclidean"):
#     pred_points = np.argwhere(pred > 0)
#     true_points = np.argwhere(true > 0)
#
#     if len(pred_points) == 0 or len(true_points) == 0:
#         return np.nan
#
#     # 计算从预测到真实标签的最小距离
#     forward_distances = cdist(pred_points, true_points, metric=distance).min(axis=1)
#     # 计算从真实标签到预测的最小距离
#     backward_distances = cdist(true_points, pred_points, metric=distance).min(axis=1)
#
#     # 合并所有的距离，并计算 95% 分位数
#     all_distances = np.concatenate([forward_distances, backward_distances])
#     hd95_value = np.percentile(all_distances, 95)
#     return hd95_value
def print_iou(iou, dname='voc'):
    iou_dict = {}
    for i in range(len(iou)-1):
        iou_dict[i] = iou[i+1]
    print(iou_dict)

    return iou_dict

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist
def dice_coeff(pred,target):
    smooth = 1e-5
    m1=pred.flatten()
    m2=target.flatten()
    intersection=(m1*m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
# def scores(label_trues, label_preds, n_class):
#     hist = np.zeros((n_class, n_class))
#     for lt, lp in zip(label_trues, label_preds):
#         hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
#     acc = np.diag(hist).sum() / hist.sum()
#     acc_cls = np.diag(hist) / hist.sum(axis=1)
#     acc_cls = np.nanmean(acc_cls)
#     iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
#     valid = hist.sum(axis=1) > 0  # added
#     mean_iu = np.nanmean(iu[valid])
#     freq = hist.sum(axis=1) / hist.sum()
#     fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
#     cls_iu = dict(zip(range(n_class), iu))
#
#     return {
#         "Pixel Accuracy": acc,
#         "Mean Accuracy": acc_cls,
#         "Frequency Weighted IoU": fwavacc,
#         "Mean IoU": mean_iu,
#         "Class IoU": cls_iu,
#     }
# def scores(label_trues, label_preds, n_class):
#     hist = np.zeros((n_class, n_class))
#     dice_scores = []
#
#     for lt, lp in zip(label_trues, label_preds):
#         hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
#         # 计算 Dice 系数
#         dice = dice_coeff(lt, lp)
#         dice_scores.append(dice)
#
#     acc = np.diag(hist).sum() / hist.sum()
#     acc_cls = np.diag(hist) / hist.sum(axis=1)
#     acc_cls = np.nanmean(acc_cls)
#     iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
#     valid = hist.sum(axis=1) > 0  # added
#     mean_iu = np.nanmean(iu[valid])
#     freq = hist.sum(axis=1) / hist.sum()
#     fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
#     cls_iu = dict(zip(range(n_class), iu))
#
#     return {
#         "Pixel Accuracy": acc,
#         "Mean Accuracy": acc_cls,
#         "Frequency Weighted IoU": fwavacc,
#         "Mean IoU": mean_iu,
#         "Class IoU": cls_iu,
#         "mean Dice Scores": np.array(dice_scores).mean(),  # 添加 Dice 系数
#     }
def iou_score(predict: np.ndarray, label: np.ndarray):
    # 确保是二值化的，背景为0，目标区域为1

    # 计算交集和并集
    intersection = np.sum(predict * label)
    union = np.sum(predict) + np.sum(label) - intersection

    # 计算 IoU，避免除以零
    if union == 0:
        return np.nan  # 或者返回 0, 具体根据需求调整

    return intersection / union
def scores(label_trues, label_preds, eval_list, n_class):
    hist = np.zeros((n_class, n_class))
    sample_hist = defaultdict(lambda: np.zeros((n_class, n_class)))
    dice_scores = []
    hd95_scores = []  # 用于记录每对预测和真实标签的 HD95 值
    sample_dice_scores = defaultdict(list)  # 按样例存储 Dice 得分
    sample_hd95_scores = defaultdict(list)  # 按样例存储 HD95 得分
    sample_iou_scores = defaultdict(list)
    # 对每个样例的 hist 计算各类指标
    sample_acc = []
    sample_acc_cls = []
    sample_mean_iu = []
    sample_freq_wavacc = []
    for lt, lp, lpath in zip(label_trues, label_preds, eval_list):
        sample_name = lpath.split('/')[-2]
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
        sample_hist[sample_name] += _fast_hist(lt.flatten(), lp.flatten(), n_class)

        # 计算 Dice 系数
        dice = dice_coeff(lt, lp)
        dice_scores.append(dice)
        sample_dice_scores[sample_name].append(dice)

        ioutmp = iou_score(lt,lp)
        sample_iou_scores[sample_name].append(ioutmp)

        # 使用 MedPy 计算 HD95
        try:
            hd95_value = hd95(lp, lt)
        except:
            hd95_value = 344
        hd95_scores.append(hd95_value)
        sample_hd95_scores[sample_name].append(hd95_value)


    for sample_hist_val in sample_hist.values():
        acc = np.diag(sample_hist_val).sum() / sample_hist_val.sum()
        sample_acc.append(acc)

        acc_cls = np.diag(sample_hist_val) / sample_hist_val.sum(axis=1)
        sample_acc_cls.append(np.nanmean(acc_cls))

        iu = np.diag(sample_hist_val) / (sample_hist_val.sum(axis=1) + sample_hist_val.sum(axis=0) - np.diag(sample_hist_val))
        valid = sample_hist_val.sum(axis=1) > 0
        sample_mean_iu.append(np.nanmean(iu[valid]))

        freq = sample_hist_val.sum(axis=1) / sample_hist_val.sum()
        sample_freq_wavacc.append((freq[freq > 0] * iu[freq > 0]).sum())

    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()

    # 对每个样例的得分进行平均计算
    mean_dice_scores = {k: np.nanmean(v) for k, v in sample_dice_scores.items()}
    mean_hd95_scores = {k: np.nanmean(v) for k, v in sample_hd95_scores.items()}
    mean_iou_scores = {k: np.nanmean(v) for k, v in sample_iou_scores.items()}
    # 计算所有样例的平均 Dice 和 HD95
    overall_mean_dice = np.nanmean(list(mean_dice_scores.values()))
    overall_mean_hd95 = np.nanmean(list(mean_hd95_scores.values()))
    overall_mean_iou = np.nanmean(list(mean_iou_scores.values()))

    return {
        # "Pixel Accuracy": acc,
        # "Mean Accuracy": acc_cls,
        # "Mean IoU": mean_iu,
        # "mean Dice Scores": np.array(dice_scores).mean(),
        # "mean HD95": np.nanmean(hd95_scores),
        "Sample-wise Mean Dice Scores": overall_mean_dice,
        "Sample-wise Mean HD95": overall_mean_hd95,
        # "Sample-wise Pixel Accuracy": np.nanmean(sample_acc),
        # "Sample-wise Mean Accuracy": np.nanmean(sample_acc_cls),
        "Sample-wise Mean IoU": overall_mean_iou,
        # "Sample-wise Frequency Weighted IoU": np.nanmean(sample_freq_wavacc),
    }
def run_eval_cam(args, print_log=True, is_coco=False):
    preds = []
    labels = []
    n_images = 0
    for i, id in enumerate(eval_list):
        n_images += 1
        if args.cam_type == 'png':
            label_path = os.path.join(args.cam_out_dir, id + '.png')
            cls_labels = np.asarray(Image.open(label_path), dtype=np.uint8)
        else:
            # cam_dict = np.load(os.path.join(args.cam_out_dir, id.split('/')[-1].split('.')[0] + '.npy'), allow_pickle=True).item()
            # cam_dict = np.load(os.path.join(args.cam_out_dir, id.split('/')[-1].split('.')[0] + '.npz'),
            #                    allow_pickle=True)
            cam_dict = np.array(Image.open(id).resize((240, 240)))
            cam_dict [cam_dict>0]=1
            cams = cam_dict
            # if 'bg' not in args.cam_type:
            #     if args.cam_eval_thres < 1:
            #         cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
            #     else:
            #         bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), args.cam_eval_thres)
            #         cams = np.concatenate((bg_score, cams), axis=0)
            # keys = np.array([0,1])
            # cls_labels = np.argmax(cams, axis=0)
            # cls_labels = keys[cls_labels].astype(np.uint8)
            cls_labels = cams
        preds.append(cls_labels)
        gt_file = os.path.join(args.gt_root, id.split('/')[-1].split('.')[0]+ '.npz')
        gt = np.array(np.load(gt_file)['arr_0']).astype(np.uint8)
        gt[gt > 0] = 1
        labels.append(gt)

    iou = scores(labels, preds, eval_list,n_class=2)

    if print_log:
        print(iou)
    return iou["Sample-wise Mean IoU"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_out_dir", default="./cam_out", type=str)
    parser.add_argument("--cam_type", default="attn_highres", type=str)
    parser.add_argument('--img_root', type=str, default='/root/data1/brats/val/yes')
    parser.add_argument("--split_file", default="/home/xxx/datasets/VOC2012/ImageSets/Segmentation/train.txt", type=str)
    parser.add_argument("--cam_eval_thres", default=2, type=float)
    parser.add_argument("--gt_root", default="/home/xxx/datasets/VOC2012/SegmentationClassAug", type=str)
    args = parser.parse_args()

    is_coco = 'coco' in args.cam_out_dir

    # eval_list = os.listdir(args.img_root)
    # eval_list = [
    #     '/root/data1/brats/val/valyes/' + x.split('.')[0].split('_')[0] + '_' + x.split('.')[0].split('_')[
    #         1] + '/' + x.split('.')[0] + '.png' for x in eval_list]
    root_folder = "/root/data1/brats/BTATS/output/perseg/root/data1/brats/BTATS/perseg/ref_composed.txt_x_x_x_SD_0.1_[0.3, 0.2, 0.1]_erosion_32_4_32_4"

    # 初始化存储文件名的列表
    eval_list = []

    # 遍历根文件夹下的所有文件夹和文件
    for root, dirs, files in os.walk(root_folder):
        for file_name in files:
            # 判断文件名是否以 "BraTs" 开头
            if file_name.split('_')[0] == "BraTS2021":
                eval_list.append(file_name)
    eval_list = [
        '/root/data1/brats/BTATS/output/perseg/root/data1/brats/BTATS/perseg/ref_composed.txt_x_x_x_SD_0.1_[0.3, 0.2, 0.1]_erosion_32_4_32_4/'
        + x.split('.')[0].split('_')[0] + '_' + x.split('.')[0].split('_')[
            1] + '/' + x.split('.')[0] + '.jpg' for x in eval_list]
    if 'bg' in args.cam_type or 'png' in args.cam_type:
        iou = run_eval_cam(args, True)
    else:
        # if args.cam_eval_thres < 1:
        #     thres_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        # else:
            # if 'attn' in args.cam_type:
        thres_list = [1]
            # else:
            #     thres_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        max_iou = 0
        max_thres = 0
        for thres in thres_list:
            args.cam_eval_thres = thres
            iou = run_eval_cam(args, print_log=False, is_coco=is_coco)
            print(thres, iou)
            if iou > max_iou:
                max_iou = iou
                max_thres = thres

        args.cam_eval_thres = max_thres
        iou = run_eval_cam(args, print_log=True, is_coco=is_coco)

    # args.cam_eval_thres = max_thres
        # iou = run_eval_cam(args, print_log=True, is_coco=is_coco)
# --cam_out_dir /root/data1/CLIP-ES-main/output/brats/cams --cam_type attn_highres --gt_root /root/data1/brats/val/label --split_file ./voc12/train.txt
# --cam_out_dir /root/data1/cam-MEDSAM/work_dir/最终/twosam_addcam_1033 --cam_type caa_sam1_stack_pred --gt_root /root/data1/brats/val/label --split_file ./voc12/train.txt