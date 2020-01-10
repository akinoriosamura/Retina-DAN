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

    print(sys.argv)
except:
    print("error: please writing read image dir path and save dir path")

# initialize
print("======= start initililzing ==============")
img_paths = get_img_paths(im_dir_path)

detector = RetinaDetector()
alignmentor = DanAlignmentor()
label_j = {}
save_img_path = os.path.join(save_dir_path, 'img')
os.makedirs(save_img_path, exist_ok=True)
save_labeled_path = os.path.join(save_dir_path, 'labeled')
os.makedirs(save_labeled_path, exist_ok=True)
save_json_path = os.path.join(save_dir_path, 'json')
os.makedirs(save_json_path, exist_ok=True)

for img_id, img_path in enumerate(img_paths):

    fname = os.path.basename(img_path)
    color_img = cv2.imread(img_path, 1)
    # DAN landmarkがgray_imageのみ受け付ける
    if color_img is None:
        print("img is None")
        continue
    if len(color_img.shape) > 2:
        gray_img = np.mean(color_img, axis=2).astype(np.uint8)
    else:
        gray_img = color_img.astype(np.uint8)

    print("==== start detector ====")
    bbs = detector.detect(color_img)
    if bbs is not None:
        for bb_id, bb in enumerate(bbs):
            print("==== start alignment ====")
            landmarks = alignmentor.alignment(bb, gray_img)
            if landmarks is not None:
                # get new labels for crop image
                bb_x = int(bb[0])
                bb_y = int(bb[1])
                bb_w = int(bb[2] - bb[0])
                bb_h = int(bb[3] - bb[1])

                # set image
                each_fname = fname[:-4] + "_" + str(bb_id) + ".jpg"
                color_labeled_image = color_img.copy()
                label_j[each_fname] = {}
                label_j[each_fname]["bb"] = {}
                label_j[each_fname]["landmark"] = {}

                # set bb label in img 
                cv2.rectangle(color_labeled_image, (bb_x, bb_y), (bb_x+bb_w, bb_y+bb_h), (255, 0, 0))
                # set bb label in json 
                label_j[each_fname]["bb"]["left"] = bb_x
                label_j[each_fname]["bb"]["top"] = bb_y
                label_j[each_fname]["bb"]["width"] = bb_w
                label_j[each_fname]["bb"]["height"] = bb_h

                # set landmark label
                for i in range(landmarks.shape[0]):
                    land_x = int(landmarks[i, 0])
                    land_y = int(landmarks[i, 1])
                    # set landmark label in img 
                    cv2.circle(color_labeled_image, (land_x, land_y), 4, (0, 255, 0), 4)
                    # set label to json
                    label_j[each_fname]["landmark"][str(i)] = {}
                    label_j[each_fname]["landmark"][str(i)]["x"] = land_x
                    label_j[each_fname]["landmark"][str(i)]["y"] = land_y
                    # import pdb; pdb.set_trace()
    
                # save image
                cv2.imwrite(os.path.join(save_img_path, each_fname), color_img)
                # save labeled image
                lfname = each_fname[:-4] + ".labeled.jpg"
                cv2.imwrite(os.path.join(save_labeled_path, lfname), color_labeled_image)

    if img_id % 100 == 0 and img_id != 0:
        print("=========== process num ============: ", img_id)


# save label json
jname = "segment_moru_dataset.json"
with open(os.path.join(save_json_path, jname), "w") as f:
    json.dump(label_j, f, indent=4, ensure_ascii=False)
