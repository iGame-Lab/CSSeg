import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import torch.nn.functional as F
import joblib
import multiprocessing
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import cv2
from PIL import Image
import argparse


class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def crf(n_jobs,image,label,cam,mean_bgr):
    """
    CRF post-processing on pretrained_models-computed logits
    """

    # Configuration
    # torch.set_grad_enabled(False)
    # print("# jobs:", n_jobs)

    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

    # Process per sample
    def process(image,label,cam,mean_bgr=None):
        image = image.astype(np.float32)
        gt_label = np.asarray(label, dtype=np.int32)
        # Mean subtraction
        image -= mean_bgr
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        cams=cam
        cams = np.expand_dims(cam,axis=0)

        bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), 1)
        cams = np.concatenate((bg_score, cams), axis=0)
        prob = cams

        image = image.astype(np.uint8).transpose(1, 2, 0)
        prob = postprocessor(image, prob)

        label = np.argmax(prob, axis=0)


        confidence = np.max(prob, axis=0)
        label[confidence < 0.95] = 1

        return label.astype(np.uint8)

    return process(image,label,cam,mean_bgr)


    # CRF in multi-process
