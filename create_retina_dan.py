import numpy as np
import os
import json
import cv2
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'RetinaFace'))
from retina_detector import RetinaDetector
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'DAN', 'DeepAlignmentNetwork'))
from dan_alignmentor import DanAlignmentor


def get_img_paths(im_dir_path):
    img_paths = []
    for curDir, dirs, files in os.walk(im_dir_path):
        for file in files:
            if (".jpg" in file) or (".png" in file) or (".JPG" in file) or (".PNG" in file):
                img_paths.append(os.path.join(curDir, file))

    print("num of imgs: ", len(img_paths))

    return img_paths

try:
    im_dir_path = sys.argv[1]
    save_dir_path = sys.argv[2]
except:
    print("error: please writing read image dir path and save dir path")

# initialize
print("======= start initililzing ==============")
img_paths = get_img_paths(im_dir_path)
detector = RetinaDetector()
alignmentor = DanAlignmentor()
label_j = {}
save_imgs_path = os.path.join(save_dir_path, 'images')
os.makedirs(save_imgs_path, exist_ok=True)
save_json_path = os.path.join(save_dir_path, 'json')
os.makedirs(save_json_path, exist_ok=True)

for img_path in img_paths:
    color_img = cv2.imread(img_path, 1)
    # DAN landmarkがgray_imageのみ受け付ける
    if len(color_img.shape) > 2:
        gray_img = np.mean(color_img, axis=2).astype(np.uint8)
    else:
        gray_img = color_img.astype(np.uint8)

    print("==== start detector ====")
    bbs = detector.detect(color_img)
    if bbs is not None:
        for bb in bbs:
            print("==== start alignment ====")
            landmarks = alignmentor.alignment(bb, gray_img)
            if landmarks is not None:
                fname = os.path.basename(img_path)
                label_j[fname] = {}
                label_j[fname]["bb"] = {}
                label_j[fname]["landmark"] = {}
                bb = bb.astype(np.int32)
                # set bb label in img 
                cv2.rectangle(color_img, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0))
                # set bb label in json 
                label_j[fname]["bb"]["left"] = float(bb[0])
                label_j[fname]["bb"]["top"] = float(bb[1])
                label_j[fname]["bb"]["width"] = float(bb[2] - bb[0])
                label_j[fname]["bb"]["height"] = float(bb[3] - bb[1])
                landmarks = landmarks.astype(np.int32)
                for i in range(landmarks.shape[0]):
                    # set landmark label in img 
                    cv2.circle(color_img, (landmarks[i, 0], landmarks[i, 1]), 4, (0, 255, 0), 4)
                    # set label to json
                    label_j[fname]["landmark"][str(i)] = {}
                    label_j[fname]["landmark"][str(i)]["x"] = float(landmarks[i, 0])
                    label_j[fname]["landmark"][str(i)]["y"] = float(landmarks[i, 1])
                    # import pdb; pdb.set_trace()
    
                # save labeled image
                lfname = fname + ".labeled.jpg"
                cv2.imwrite(os.path.join(save_imgs_path, lfname), color_img)

# save label json
dname = os.path.basename(im_dir_path)
jname = dname + ".json"
with open(os.path.join(save_json_path, jname), "w") as f:
    json.dump(label_j, f, indent=4)
