import numpy as np
import glob
import os
from PIL import Image
import math
import random
import json
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument('--cropped_region_coordinates_path', help="cropped_region_coordinates_path", default=None, type=str)
    return parser.parse_args()


def crop_and_resize_images(cropped_instances_mask_path, bounding_box):

    # cropped_instances_mask =  Image.open(cropped_instances_mask_path)
    # A dictionary contain folder name and unchanged part of this folder name
    common_part = {'aachen': 2, 'bochum': 1, 'bremen': 2, 'cologne': 2, 'darmstadt': 2, 'dusseldorf': 2, 'erfurt': 2, 'hamburg': 1, 'hanover': 1, 'jena': 2, 'krefeld': 1, 'monchengladbach': 1, 'strasbourg': 1, 'stuttgart': 2, 'tubingen': 2, 'ulm': 2, 'weimar': 2, 'zurich': 2}
    city = os.path.basename(cropped_instances_mask_path).split('_')[0]
    
    if common_part[city] == 1:
        unchanged_part = os.path.basename(cropped_instances_mask_path).split('_')[1]
        color_img_path = cropped_instances_mask_path.replace(city + '_syn', city).replace(unchanged_part, '000000').replace('gtFine_instanceIds', 'gtFine_color')
        save_color_img_path = cropped_instances_mask_path.replace('gtFine_instanceIds', 'gtFine_color')
        
        labelIds_img_path = cropped_instances_mask_path.replace(city + '_syn', city).replace(unchanged_part, '000000').replace('gtFine_instanceIds', 'gtFine_labelIds')
        save_labelIds_img_path = cropped_instances_mask_path.replace('gtFine_instanceIds', 'gtFine_labelIds')
        
        # Because unchanged part of some folder have value 000000 or 000001
        if not os.path.exists(color_img_path):
            color_img_path = cropped_instances_mask_path.replace(city + '_syn', city).replace(unchanged_part, '000001').replace('gtFine_instanceIds', 'gtFine_color')
            labelIds_img_path = cropped_instances_mask_path.replace(city + '_syn', city).replace(unchanged_part, '000001').replace('gtFine_instanceIds', 'gtFine_labelIds')
        
    elif common_part[city] == 2:
        unchanged_part = os.path.basename(cropped_instances_mask_path).split('_')[2]
        color_img_path = cropped_instances_mask_path.replace(city + '_syn', city).replace(unchanged_part, '000019').replace('gtFine_instanceIds', 'gtFine_color')
        save_color_img_path = cropped_instances_mask_path.replace('gtFine_instanceIds', 'gtFine_color')
        
        labelIds_img_path = cropped_instances_mask_path.replace(city + '_syn', city).replace(unchanged_part, '000019').replace('gtFine_instanceIds', 'gtFine_labelIds')
        save_labelIds_img_path = cropped_instances_mask_path.replace('gtFine_instanceIds', 'gtFine_labelIds')
        
    ## For color image 
    color_img = Image.open(color_img_path)
    crop_color_img = color_img.crop((bounding_box)).resize((2048, 1024), resample=Image.NEAREST)
    crop_color_img.save(save_color_img_path)
    
    ## For labelIds image 
    labelIds_img = Image.open(labelIds_img_path)
    crop_labelIds_img = labelIds_img.crop((bounding_box)).resize((2048, 1024), resample=Image.NEAREST)
    crop_labelIds_img.save(save_labelIds_img_path)

    
if __name__ == "__main__":
    
    args = parse_args()
    cropped_region_coordinates_path =  args.cropped_region_coordinates_path 
    
    with open(cropped_region_coordinates_path, 'r') as file:
        cropped_region_coordinates = json.load(file)

    for i in tqdm(range(0, len(cropped_region_coordinates))):
        bounding_box = cropped_region_coordinates[i]["bbox_coordinate"] 
        cropped_instances_mask_path = cropped_region_coordinates[i]["cropped_instance_mask"] 
        crop_and_resize_images(cropped_instances_mask_path, bounding_box)
        
