# retina-DAN
get almost correct landmark labeled by
 - face detection  
 RetinaFace
 - face landmark  
 DAN

## requirements
 - install pipenv
 - set process data in `data/`
 - change data path
 chage process dir name in Pipfile's `scripts` 
 - download RetinaFace pretrained model  
 Pretrained Model: RetinaFace-R50 ([baidu cloud](https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ) or [dropbox](https://www.dropbox.com/s/53ftnlarhyrpkg2/retinaface-R50.zip?dl=0)) is a medium size model with ResNet50 backbone.  
 It can output face bounding boxes and five facial landmarks in a single forward pass.  
 WiderFace validation mAP: Easy 96.5, Medium 95.6, Hard 90.4.  
 - unzip and set the RetinaFace pretrained model in `src/RetinaFace/model/`
 - download DAN pretrained model `DAN-Menpo-tracking.npz`  
 the model available on Dropbox [here](https://www.dropbox.com/sh/v754z1egib0hamh/AADGX1SE9GCj4h3eDazsc0bXa?dl=0) or Google drive [here](https://drive.google.com/open?id=168tC2OxS5DjyaiuDy_JhIV3eje8K_PLJ).  
 - set `DAN-Menpo-tracking.npz` in `src/DAN/data/`
 - setup
 ```
 pipenv install
 pipenv shell
 sh setup.sh
 exit
 ```

## run
```
pipenv run alignment [img dir path] [save dir path]
```

## output
get labeled image in `[save dir path]/images/` and labeled json in `[save dir path]/json/`