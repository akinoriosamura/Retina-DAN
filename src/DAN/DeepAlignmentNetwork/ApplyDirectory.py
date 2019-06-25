from FaceAlignment import FaceAlignment
import numpy as np
import os
import cv2
import utils
import sys
from mtcnn.mtcnn import MTCNN

try:
    im_directory_path = sys.argv[1]
    # xml_path = sys.argv[2]
except:
    print("error: please write image directory path")

def get_img_path(im_directory_path):
    img_path = im_directory_path
    return img_path

def get_bbs(detector, img):
    # rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50)) 
    rects = detector.detect_faces(img)
    bbs = []
    print("rects: ", rects)
    if len(rects) > 0:
        for rect in rects:
            if rect["confidence"] < 0.8:
                continue
            bb = {}
            bb["minX"] = rect["box"][0]
            bb["maxX"] = rect["box"][0] + rect["box"][2]
            bb["minY"] = rect["box"][1]
            bb["maxY"] = rect["box"][1] + rect["box"][3]

            bbs.append(bb)

        print("bbs: ", bbs)
        return bbs
    else:
        return None

def get_landmarks(bb, img, model):
    landmarks = None 
    print("bb: ", bb)
    initLandmarks = utils.bestFitRect(None, model.initLandmarks, [bb["minX"], bb["minY"], bb["maxX"], bb["maxY"]])
    landmarks, confidence = model.processImg(img[np.newaxis], initLandmarks)
    print("conf: ", confidence)

    if confidence < 0.3:
        landmarks = None

    return landmarks

# (width, height, channels, DAN num, confidence値の層（トラッキングでどちらを優先するかに）
model = FaceAlignment(112, 112, 1, 1, True)
# model.loadNetwork("../data/DAN-Menpo.npz")
model.loadNetwork("../data/DAN-Menpo-tracking.npz")

img_path = get_img_path(im_directory_path)
# cascade = cv2.CascadeClassifier("../data/haarcascade_frontalface_alt.xml")
detector = MTCNN()

print ("Press space to detect the face, press escape to exit")
color_img = cv2.imread(img_path, 1)
# landmarkがgray_imageのみ受け付ける
if len(color_img.shape) > 2:
    gray_img = np.mean(color_img, axis=2).astype(np.uint8)
else:
    gray_img = color_img.astype(np.uint8)

print("img type: ", type(color_img))
print("img: ", color_img.shape)
print("img type: ", type(gray_img))
print("img: ", gray_img.shape)

bbs = get_bbs(detector, color_img)
if bbs is not None:
    for bb in bbs:
        landmarks = get_landmarks(bb, gray_img, model)
        if landmarks is not None:
            landmarks = landmarks.astype(np.int32)
            cv2.rectangle(color_img, (bb["minX"], bb["minY"]), (bb["maxX"], bb["maxY"]), (255, 0, 0))
            for i in range(landmarks.shape[0]):
                cv2.circle(color_img, (landmarks[i, 0], landmarks[i, 1]), 4, (0, 255, 0), 4)


cv2.imwrite("./labeld.jpg", color_img)
# cv2.imshow("image", color_img)
# key = cv2.waitKey(0)
