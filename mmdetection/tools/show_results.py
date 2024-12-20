from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import torch
from collections import OrderedDict
import pandas as pd 
import numpy as np 
from splits import get_unseen_class_ids, get_seen_class_ids
import os
from mmdet.datasets import build_dataset
from mmdet.apis.runner import copy_synthesised_weights 
from mmcv import Config

# config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
config_file = 'configs/faster_rcnn_r101_fpn_1x.py'
checkpoint_file = 'work_dirs/coco2014/epoch_12_0.pth'
# checkpoint_file = 'cp_48_17/epoch_12.pth'
syn_weights = '../checkpoints/coco_65_15/pth12_gtbgno_/classifier_80_6.pth'
#'/raid/mun/codes/zero_shot_detection/cvpr18xian_pascal_voc/checkpoints/coco_48_17/classifier_best.pth'
# syn_weights = '/raid/mun/codes/zero_shot_detection/cvpr18xian_pascal_voc/checkpoints/coco/classifier_latest.pth'
# build the model from a config file and a checkpoint file
# config_file = 'configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py'
# checkpoint_file = 'work_dirs/faster_rcnn_r101_fpn_1x_voc0712/epoch_4.pth'
# syn_weights = '/raid/mun/codes/zero_shot_detection/cvpr18xian_pascal_voc/checkpoints/classifier_best.pth'
score_thr = 0.99
try:
    os.makedirs('det_results/coco')
    os.makedirs(f'det_results/coco_{score_thr}')
    os.makedirs('det_results/voc')
except OSError:
    pass

# import pdb; pdb.set_trace()

model = init_detector(config_file, checkpoint_file, device='cuda')
cfg = Config.fromfile(config_file)
dataset = build_dataset(cfg.data.test, {'test_mode': True})
# copy_syn_weights(syn_weights, model)
copy_synthesised_weights(model, syn_weights, 'coco', split='65_15')
root = 'data/coco2014/val2014'
# df = pd.read_csv('../MSCOCO/validation_coco_unseen_all.csv', header=None)
# file_names = np.unique(df.iloc[:, 0].values)
# files_path = [f"{root}{file_name}" for file_name in file_names]
# files_path = np.array(files_path)
# img_infos
# for idx, img in enumerate(files_path[:1000]):
# import pdb; pdb.set_trace()
import random
# color = "%06x" % random.randint(0, 0xFFFFFF)
from splits import COCO_ALL_CLASSES
color_maps = {label: (random.randint(0, 255), random.randint(120, 255), random.randint(200, 255)) for label in COCO_ALL_CLASSES}
# det_results = mmcv.load('gen_coco_results.pkl')
det_results = mmcv.load('coco_results.pkl')
gen_filenames = [
'COCO_val2014_000000008676.jpg',
'COCO_val2014_000000012827.jpg',  'COCO_val2014_000000056430.jpg',  'COCO_val2014_000000403817.jpg',
'COCO_val2014_000000483108.jpg',
'COCO_val2014_000000012085.jpg',  'COCO_val2014_000000027371.jpg',  'COCO_val2014_000000069411.jpg',
'COCO_val2014_000000428454.jpg',
'COCO_val2014_000000553721.jpg']

zsd = [
    'COCO_val2014_000000052066.jpg',  'COCO_val2014_000000058225.jpg' ,
    'COCO_val2014_000000128644.jpg' , 'COCO_val2014_000000350073.jpg' ,'COCO_val2014_000000519299.jpg',
    'COCO_val2014_000000054277.jpg',  'COCO_val2014_000000101088.jpg' , 'COCO_val2014_000000171058.jpg',  
    'COCO_val2014_000000512455.jpg'  ,'COCO_val2014_000000572517.jpg',
]
start = 0
img_infos = dataset.img_infos
# [start:start+200]
# [:1000]
# import pdb; pdb.set_trace()
for idx, info in enumerate(img_infos):
    img = f"{root}/{info['filename']}"
    if info['filename'] in gen_filenames:
        result = inference_detector(model, img)
        # result = det_results[start+idx]#inference_detector(model, img)
        # import pdb; pdb.set_trace()
        out_file = f"det_results/coco_{score_thr}/{img.split('/')[-1]}"
        show_result(f"{img}", result, model.CLASSES, out_file=out_file,show=False, score_thr=score_thr, dataset='coco')
        print(f"[{idx:03}/{len(img_infos)}]")
    
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
# copy_syn_weights(syn_weights, model)
# copy_synthesised_weights(model, syn_weights)

# root = '/raid/mun/codes/data/pascalv_voc/VOCdevkit/'
# df = pd.read_csv('../VOC/testval_voc07_unseen.csv', header=None)
# file_names = np.unique(df.iloc[:, 0].values)
# files_path = [f"{root}{file_name[14:]}" for file_name in file_names]
# files_path = np.array(files_path)
# for idx, img in enumerate(files_path[:100]):

#     result = inference_detector(model, img)
#     out_file = f"det_results/voc/{img.split('/')[-1]}"
#     show_result(img, result, model.CLASSES, out_file=out_file,show=False, score_thr=0.3)
#     print(f"[{idx:03}/{len(files_path)}]")

# ./tools/dist_test.sh configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py work_dirs/faster_rcnn_r101_fpn_1x_voc0712/epoch_4.pth 8 --syn_weights /raid/mun/codes/zero_shot_det
# ection/cvpr18xian_pascal_voc/checkpoints/VOC/classifier_best.pth --out voc_detections.p