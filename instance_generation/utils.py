import os
import json
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F
import cv2
from tqdm import tqdm

#def is_exist(input_image_path, output_folder, suffix="BlendedDiff", i=4):
#    input_name = "_".join(input_image_path.split("/")[-1].split(".")[0].split("_")[:4] + [suffix] + ["_{i:06d}".format(i=i)])
#    return any(input_name in file for file in os.listdir(output_folder))

def is_exist(input_image_path, output_folder, suffix="BlendedDiffusion"):
    input_name = "_".join(input_image_path.split("/")[-1].split(".")[0].split("_")[:3] + [suffix])
    return any(input_name in file for file in os.listdir(output_folder))

def get_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        return json.load(f)

def get_cities(folder_path):
    try:
        # Only include files that do not contain an underscore
        cities = [file for file in os.listdir(folder_path) if "_" not in file]
        return cities
    except FileNotFoundError:
        print(f"The {folder_path} folder does not exist.")
        return []  # Return an empty list if the folder cannot be found

def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch

def get_clip_feature(model, processor, input, is_image=False):
    which_layer_text = 'before'
    which_layer_image = 'after_reproject'

    if is_image:
        if input is None:
            return None
        image = Image.open(input).convert("RGB")
        inputs = processor(images=[image], return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].cuda()
        inputs['input_ids'] = torch.tensor([[0, 1, 2, 3]]).cuda()
        outputs = model(**inputs)
        feature = outputs.image_embeds 
        if which_layer_image == 'after_reproject':
            feature = project(feature, torch.load('projection_matrix').cuda().T).squeeze(0)
            feature = (feature / feature.norm()) * 28.7 
            feature = feature.unsqueeze(0)
    else:
        if input is None:
            return None
        inputs = processor(text=input, return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = torch.ones(1, 3, 224, 224).cuda()
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)
        feature = outputs.text_model_output.pooler_output
    return feature

def get_masks(masks, batch=1):
    inpainting_mask = Image.open(masks).resize((64, 64))
    inpainting_mask = np.array(inpainting_mask)
    inpainting_mask = torch.Tensor(inpainting_mask).unsqueeze(0).cuda()
    inpainting_mask[inpainting_mask > 0] = 255
    inpainting_mask = 1 - (inpainting_mask / 255)
    inpainting_mask = inpainting_mask.unsqueeze(0).repeat(batch, 1, 1, 1)
    return inpainting_mask

def complete_mask(has_mask, max_objs):
    mask = torch.ones(1, max_objs)
    if has_mask is None:
        return mask
    if isinstance(has_mask, (int, float)):
        return mask * has_mask
    for idx, value in enumerate(has_mask):
        mask[0, idx] = value
    return mask

def mask2box(mask):
    mask = Image.open(mask)
    mask = np.array(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x + w, y + h])
    image_height, image_width = mask.shape
    normalized_boxes = [
        [round(x_min / image_width, 2), round(y_min / image_height, 2), round(x_max / image_width, 2), round(y_max / image_height, 2)]
        for x_min, y_min, x_max, y_max in boxes
    ]
    return normalized_boxes