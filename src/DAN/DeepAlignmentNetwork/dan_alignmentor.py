import numpy as np
import os
import cv2
import utils
import sys

from FaceAlignment import FaceAlignment


class DanAlignmentor(object):
    def __init__(self):
        # (width, height, channels, DAN num, confidence値の層（トラッキングでどちらを優先するかに）
        self.model = FaceAlignment(112, 112, 1, 1, True)
        self.model_path = 'data/DAN-Menpo-tracking.npz'
        self.model_path = os.path.join(os.path.dirname(__file__), '..', self.model_path)
        print("dan model path: ", self.model_path)
        self.model.loadNetwork(self.model_path)

    def alignment(self, bb, img):
        landmarks = None 
        initLandmarks = utils.bestFitRect(None, self.model.initLandmarks, [bb[0], bb[1], bb[2], bb[3]])
        landmarks, confidence = self.model.processImg(img[np.newaxis], initLandmarks)
        print("alignment conf: ", confidence)

        if confidence < 0.9:
            landmarks = None

        return landmarks
