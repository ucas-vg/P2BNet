# Introduction

TODO list:

- [x] add TinyPerson dataset and evaluation
- [x] add crop and merge for image during inference
- [x] implement RetinaNet and Faster-FPN baseline on TinyPerson
- [x] add SM/MSM experiment support
- [ ] add visDronePerson dataset support and baseline performance
- [ ] add point localization task for TinyPerson
- [ ] add point localization task for visDronePerson
- [ ] add point localization task for COCO

# Prerequisites

### [install mmdetection](./docs/install.md>)
```
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y
# conda install -c pytorch pytorch=1.5.0 cudatoolkit=10.2 torchvision -y
# install the latest mmcv
pip install mmcv-full --user
# install mmdetection

pip uninstall pycocotools   # sometimes need to source deactivate before, for 
pip install -r requirements/build.txt
pip install -v -e . --user  # or try "python setup.py develop" if get still got pycocotools error
chmod +x tools/dist_train.sh
```

```
conda install scikit-image  # or pip install scikit-image
```

- [note]: if your need to modified from origin mmdetection code, see [here](docs/tov/code_modify.md), otherwise do not need any other modified.
- [note]: for more about evaluation, see [evaluation_of_tiny_object.md](docs/tov/evaluation_of_tiny_object.md)
### prepare dataset
#### TinyPerson

to train baseline of TinyPerson, download the mini_annotation of all annotation is enough, 
which can be downloaded as tiny_set/mini_annotations.tar.gz in [Baidu Yun(password:pmcq) ](https://pan.baidu.com/s/1kkugS6y2vT4IrmEV_2wtmQ)/
[Google Driver](https://drive.google.com/open?id=1KrH9uEC9q4RdKJz-k34Q6v5hRewU5HOw).

```
mkdir data
ln -s ${Path_Of_TinyPerson} data/tiny_set
tar -zxvf data/tiny_set/mini_annotations.tar.gz && mv mini_annotations data/tiny_set/
```

#### COCO

```
ln -s ${Path_Of_COCO} data/coco
```

# Experiment
## TinyPerson

For running more experiment, to see [bash script](configs2/TinyPerson/base/Baseline_TinyPerson.sh)

```shell script
# exp1.2: Faster-FPN, 2GPU
export GPU=2 && LR=0.01 && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/TinyPerson/base/faster_rcnn_r50_fpn_1x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_fpn_1x_TinyPerson640/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# Scale Match
# exp4.0 coco-sm-tinyperson: Faster-FPN, coco batch=8x2
export GPU=2 && export LR=0.01 && export BATCH=8 && export CONFIG="faster_rcnn_r50_fpn_1x_coco_sm_tinyperson"
export COCO_WORK_DIR="../TOV_mmdetection_cache/work_dir/TinyPerson/scale_match/${CONFIG}/lr${LR}_1x_${BATCH}b${GPU}g/"
CUDA_VISIBLE_DEVICES=0,1 PORT=10001 tools/dist_train.sh configs2/TinyPerson/scale_match/${CONFIG}.py $GPU \
  --work-dir ${COCO_WORK_DIR} --cfg-options optimizer.lr=${LR} data.samples_per_gpu=${BATCH}
# python exp/tools/extract_weight.py ${COCO_WORK_DIR}/latest.pth
export TCONFIG="faster_rcnn_r50_fpn_1x_TinyPerson640"
export GPU=2 && LR=0.01 && CUDA_VISIBLE_DEVICES=0,1 PORT=10000 tools/dist_train.sh configs2/TinyPerson/base/${TCONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/scale_match/${TCONFIG}/cocosm_old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} load_from=${COCO_WORK_DIR}/latest.pth
```

- GPU: 3080 x 2
- Adap RetainaNet-c means use clip grad while training.
- COCO val $mmap$ only use for debug, cause val also add to train while sm/msm coco to TinyPerson

detector | type | $AP_{50}^{tiny}$| script | COCO200 val $mmap$ | coco batch/lr
--- | --- | ---| ---| ---| ---
Faster-FPN | - |  ~~47.90~~<br/>49.81 | exp/Baseline_TinyPerson.sh:exp1.2 | - | -
Faster-FPN | SM | ~~50.06~~<br/> | exp/Baseline_TinyPerson.sh:exp4.0 | 18.9 | 8x2/0.01
Faster-FPN | SM | ~~49.53~~<br/> | exp/Baseline_TinyPerson.sh:exp4.1 | 18.5 | 4x2/0.01
Faster-FPN | MSM | ~~49.39~~<br/> | exp/Baseline_TinyPerson.sh:exp4.2 | 12.1 | 4x2/0.01
--| --| --
Adap RetainaNet-c | -   | ~~43.66~~<br/> | exp/Baseline_TinyPerson.sh:exp2.3 | - | -
Adap RetainaNet-c | SM  | ~~50.07~~<br/> | exp/Baseline_TinyPerson.sh:exp5.1 | 19.6 | 4x2/0.01
Adap RetainaNet-c | MSM | ~~48.39~~<br/> | exp/Baseline_TinyPerson.sh:exp5.2 | 12.9 | 4x2/0.01

for more experiment, to see [TinyPerson experiment](configs2/TinyPerson/TinyPerson.md)
for detail of scale match, to see [TinyPerson Scale Match](configs2/TinyPerson/scale_match/ScaleMatch.md)

## TinyCOCO


### 2. base performance

detector | norm_grad | batch | anchor scale | quality | $COCO:mmAP$ | $COCO:mAP_{50}$ | script
--- | --- | ---| ---| ---|--- | ---|---
Faster-FPN | True | 4x4 | 2 | 75 | 14.7 | 28.9 | exp/sh/Baseline_TinyCOCO.sh:2.1.1
Faster-FPN | True | 8x4 | 2 | 75 | 14.6 | 28.9 | exp/sh/Baseline_TinyCOCO.sh:2.1.2
Faster-FPN | True | 16x4 | 2| 75 | 14.5 | 28.8 | exp/sh/Baseline_TinyCOCO.sh:2.1.3
Faster-FPN | True | 32x4 | 2 | 75 | 14.1 | 27.7 | exp/sh/Baseline_TinyCOCO.sh:2.1.4
-|-|-|-|-|-|-|-
Faster-FPN | True | 16x4 | 2  | 100 | 15.6 | 30.6 | exp/sh/Baseline_TinyCOCO.sh:2.2
Faster-FPN | False | 16x4 | 2 | 100 | **15.7** | **30.9** | exp/sh/Baseline_TinyCOCO.sh:2.3
Faster-FPN | False | 16x4 | 1.5 | 100 | 13.6 | 27.6 | exp/sh/Baseline_TinyCOCO.sh:2.3
-|-|-|-|-|-|-|-
Adap RepPoint | True | 16x4 | 2 | 100 | 18.3 | 34.0 |  exp/sh/Baseline_TinyCOCO.sh:3.1

for more experiment, to see [TinyCOCO experiment](configs2/TinyCOCO/TinyCOCO.md)

## visDronePerson

add configs(modified num_class, modified anchro scale)

experiment setting
- anchor scale is point_base_scale;
- max_per_img=-1;
- scale=1x while test
- all run in 2080Tix4

detector | anchor scale | $AP_{50}$ | $AP_{1.0}$ | script
---| ---| --- | ---| ---
RepPoint(1)       | 4 | 58.23 | 71,74 | exp/sh/Baseline_visDronePerson.sh:exp1.3
RepPoint(3)       | 2 | 61.34 | 72.93 | exp/sh/Baseline_visDronePerson.sh:exp1.1
Adap RepPoint(1)  | 2 | 62.34 | 71.40 | exp/sh/Baseline_visDronePerson.sh:exp1.2
Faster FPN(3)     | 2 | 68.97 | 76.39 | exp/sh/Baseline_visDronePerson.sh:exp2.1

for more experiment, to see [visDronePerson experiment](configs2/visDronePerson/visDronePerson.md)
