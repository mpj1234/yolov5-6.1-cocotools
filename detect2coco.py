# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2022/12/17 22:24
  @version V1.0
"""
import json
import os
import torch
from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.torch_utils import select_device

# 读取文件夹下的所有文件
input_path = './output/images'
output_path = './output'
device = ''
weights = './weights/best.pt'
imgsz = 640
source = input_path
coco_json_save = output_path + '/detect_coco.json'

data = './data/coco.yaml'
imgsz = [640, 640]
conf_thres = 0.001
iou_thres = 0.6
max_det = 100
device = ''
half = False

# 创建coco格式的预测结果
coco_json = []

# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device, data=data)
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size(imgsz, s=stride)  # check image size

# Half
half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
if pt or jit:
	model.model.half() if half else model.model.float()

# Dataloader
dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
bs = 1  # batch_size

# Run inference
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup

for path, im, im0s, vid_cap, s in dataset:
	# 获取图片名字
	image_name = os.path.basename(path).split('.')[0]

	im = torch.from_numpy(im).to(device)
	im = im.half() if half else im.float()  # uint8 to fp16/32
	im /= 255  # 0 - 255 to 0.0 - 1.0
	if len(im.shape) == 3:
		im = im[None]  # expand for batch dim

	# Inference
	pred = model(im)

	# NMS
	pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

	# Process predictions
	for i, det in enumerate(pred):  # per image
		if len(det):
			# Rescale boxes from img_size to im0 size
			det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()

			# Write results
			for *xyxy, conf, cls in reversed(det):
				# 将检测结果保存到coco_json中
				coco_json.append({
					'image_id': int(image_name),
					'category_id': int(cls) + 1,
					'bbox': [float(xyxy[0]), float(xyxy[1]), float(xyxy[2] - xyxy[0]), float(xyxy[3] - xyxy[1])],
					'score': float(conf),
					'area': float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))
				})

# 保存json文件
with open(os.path.join(coco_json_save), 'w') as f:
	# indent=2 保存json文件时，缩进2个空格
	json.dump(coco_json, f, indent=2)

print(len(coco_json), 'Done!')
