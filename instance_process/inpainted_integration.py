import os
import numpy as np
from PIL import Image
import math
import random
import json
import sys
from tqdm import tqdm
import argparse


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Process Cityscapes dataset for inpainting.')
    parser.add_argument('--inpainting_dir', help="Path to the inpainting image folder", required=True, type=str)
    parser.add_argument('--metadata_path', help="Path to the metadata file contains image paths", default="../cityscapes_synthesis/metadata.json", required=True, type=str)
    parser.add_argument('--output_folder_name', help="result folder name", default="cityscapes_diffpainting/", required=True, type=str)
    parser.add_argument('--cropped_region_coordinates', help="Path to file contains cropped coordinates of instance in image", default="../cityscapes_synthesis/cropped_region_coordinates.json", required=True, type=str) 
    
    return parser.parse_args()


def inpaint_merge(original_image_path, cropped_instance_rgb_path, inpainting_image_path, mask_image_path, bounding_box):
    """
    Merges inpainting images with the original RGB images and normalizes the inpainted region.

    Args:
        original_image_path (str): Path to the original RGB image.
        cropped_instance_rgb_path (str): Path to the cropped instance image.
        inpainting_image_path (str): Path to the inpainting image.
        mask_image_path (str): Path to the instance mask image.
        bounding_box (tuple): Bounding box of the instance.

    Returns:
        Tuple[Image, Image]: Original image merged with inpainting and its normalized version.
    """
    
    original_image = Image.open(original_image_path)
    normalized_original_image = original_image.copy()
    inpainting_image = Image.open(inpainting_image_path)
    mask_image = Image.open(mask_image_path).convert("L")

    inpainting_image = inpainting_image.resize(
        (bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]), resample=Image.NEAREST
    )
    cropped_instance_rgb = Image.open(cropped_instance_rgb_path)

    # Convert images to numpy arrays
    inpainting_array = np.array(inpainting_image)
    cropped_instance_array = np.array(cropped_instance_rgb)
    mask_array = np.array(mask_image)

    # Calculate absolute difference in masked areas
    masked_area = (mask_array == 0)
    absolute_diff = np.abs(np.subtract(inpainting_array, cropped_instance_array, dtype=int))
    masked_absolute_diff = absolute_diff[masked_area]
    mean_masked_absolute_diff = math.ceil(np.mean(masked_absolute_diff)) if len(masked_absolute_diff) != 0 else 0

    # Normalize inpainting region
    normalized_inpainting_image = Image.fromarray(
        np.abs(np.subtract(inpainting_array, mean_masked_absolute_diff, dtype=int)).astype(np.uint8)
    )

    # Merge inpainting region with the original image
    normalized_original_image.paste(normalized_inpainting_image, bounding_box, mask=mask_image)
    original_image.paste(inpainting_image, bounding_box, mask=mask_image)

    return original_image, normalized_original_image


def find_bbox(min_x, min_y, max_x, max_y, bbox_area, seperated_anot):
    """
    Finds a random bounding box for cropping based on the given area.

    Args:
        min_x, min_y, max_x, max_y (int): Coordinates of the original bounding box.
        bbox_area (int): Area of the bounding box.
        seperated_anot (Image): Separated annotation image.

    Returns:
        Tuple[int, int, int, int]: New bounding box (min_x, min_y, width, height).
    """
    mean = 3
    std_deviation = 1
    min_value = 1

    while True:
        random_value = max(min_value, math.ceil(random.normalvariate(mean, std_deviation)))
        if bbox_area * random_value <= (seperated_anot.size[0] * seperated_anot.size[1]):
            break
        
    ######################################### Condition #############################################
    # min_x_target = (min_x - target_width, min_x) => max_x_area = min_x_target + target_width      #
    # min_y_target = (min_y - target_height, min_y) => min_y_target = min_y_target + target_height  #
    # max_x_area = (max_x, image.width)                                                             #
    # max_y_area = (max_x, image.height)                                                            #
    ######################################### Condition #############################################

    target_area = random_value * bbox_area
    target_height = math.ceil((target_area / 2) ** 0.5)
    target_width = 2 * target_height

    min_x_target = random.randint(max(0, min_x - target_width), min_x)
    min_y_target = random.randint(max(0, min_y - target_height), min_y)

    while (
        min_x_target + target_width < max_x or min_x_target + target_width > seperated_anot.width
    ):
        min_x_target = random.randint(max(0, min_x - target_width), min_x)
        
        
    while (
        min_y_target + target_height < max_y or min_y_target + target_height > seperated_anot.height
    ):
        min_y_target = random.randint(max(0, min_y - target_height), min_y)

    return min_x_target, min_y_target, target_width, target_height


def crop_and_resize_images(merged_inpaint_gt_img, normalized_inpainted_merge_img, seperated_anot_path, anot_full_path):
    """
    Crops and resizes the merged inpainted images and their masks.

    Args:
        merged_inpaint_gt_img (Image): Merged inpainted image.
        normalized_inpainted_merge_img (Image): Normalized inpainted image.
        seperated_anot_path (str): Path to the separated annotation image.
        anot_full_path (str): Path to the full annotation mask.
    """
    seperated_anot = Image.open(seperated_anot_path)
    image_array = np.array(seperated_anot)
    image_array[image_array != 0] = 255
    seperated_anot = Image.fromarray(image_array.astype(np.uint8))

    non_zero_indices = seperated_anot.point(lambda p: p != 0)
    bounding_box = non_zero_indices.getbbox()

    if bounding_box is None:
        breakpoint()
        return

    min_x, min_y, max_x, max_y = bounding_box
    bb_width, bb_height = max_x - min_x, max_y - min_y
    
    # Calculate the area of the bounding box
    if bb_width > 2 * bb_height:
        bbox_area = bb_width * math.ceil(bb_width/2)
    else: bbox_area = bb_height * 2 * bb_height

    min_x_target, min_y_target, target_width, target_height = find_bbox(min_x, min_y, max_x, max_y, bbox_area, seperated_anot)
    
    # Check the coordinate do not contain less than 4 pixel to prevent the invalid polygon in anot_mask
    full_anot_mask = Image.open(anot_full_path)
    cropped_mask = full_anot_mask.crop((min_x_target, min_y_target, min_x_target + target_width, min_y_target + target_height)).resize(full_anot_mask.size, resample=Image.NEAREST)
    unique, counts= np.unique(cropped_mask, return_counts=True)
    while ((counts[unique >= 24000])<4).sum() != 0:
        min_x_target, min_y_target, target_width, target_height = find_bbox(min_x, min_y, max_x, max_y, bbox_area, seperated_anot)
        cropped_mask = full_anot_mask.crop((min_x_target, min_y_target, min_x_target + target_width, min_y_target + target_height)).resize(full_anot_mask.size, resample=Image.NEAREST)
        unique, counts= np.unique(cropped_mask, return_counts=True)

    # Check target coordinate
    filename = os.path.basename(inpainting_image_path)
    if common_part[filename.split('_')[0]] == 1:
        
        change_part = f'0{j}' +  prefix_name.split('_')[1]
        new_prefix_name = prefix_name.replace(prefix_name.split('_')[1], change_part)
        
        anot_name =  new_prefix_name + '_gtFine_instanceIds.png'
        rgb_name =  new_prefix_name + '_leftImg8bit.png'

        # For uncropped inpainted image 
        output_path_rgb_panorama = os.path.join(leftImg8bit_panorama_dir, rgb_name)
             
        # For cropped image 
        output_path_anot = os.path.join(inpainted_dir, anot_name)
        output_path_rgb = os.path.join(leftImg8bit_dir, rgb_name)
        
    elif common_part[filename.split('_')[0]] == 2:
        
        change_part = f'0{j}' +  prefix_name.split('_')[2]
        new_prefix_name = prefix_name.replace(prefix_name.split('_')[2], change_part)
        
        anot_name =  new_prefix_name + '_gtFine_instanceIds.png'
        rgb_name =  new_prefix_name + '_leftImg8bit.png'
        
        # For uncropped inpainted image 
        output_path_rgb_panorama = os.path.join(leftImg8bit_panorama_dir, rgb_name)
                
        # For cropped image 
        output_path_anot = os.path.join(inpainted_dir, anot_name)
        output_path_rgb = os.path.join(leftImg8bit_dir, rgb_name)

    # Crop mask image using the calculated dimensions and position
    full_anot_mask = Image.open(anot_full_path)
    cropped_mask = full_anot_mask.crop((min_x_target, min_y_target, min_x_target + target_width, min_y_target + target_height)).resize(full_anot_mask.size, resample=Image.NEAREST)
    cropped_mask.save(output_path_anot)
    
    ## For merged image between gt image and inpainted image
    # Crop the image using the calculated dimensions and position  
    normalized_inpainted_merge_img.save(output_path_rgb_panorama) 
    nomalized_merged_inpaint_gt_crop = normalized_inpainted_merge_img.crop((min_x_target, min_y_target, min_x_target + target_width, min_y_target + target_height)).resize(full_anot_mask.size, resample=Image.NEAREST)
    nomalized_merged_inpaint_gt_crop.save(output_path_rgb)
    
    # Save cropped images
    cropped_region_coordinates_dict.append({
        'cropped_instance_mask': str(output_path_anot),
        'bbox_coordinate': (min_x_target, min_y_target, min_x_target + target_width, min_y_target + target_height)
    })


if __name__ == "__main__":
    args = parse_args()
    inpainting_dir = args.inpainting_dir
    metadata_path = args.metadata_path
    output_folder_name = args.output_folder_name
    cropped_region_coordinates_path = args.cropped_region_coordinates
    model_type = inpainting_dir.split('/')[-2]
    ver = inpainting_dir.split('/')[-1][0] + inpainting_dir.split('/')[-1][-1]

    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    with open(cropped_region_coordinates_path, 'r') as file:
        cropped_region_coordinates = json.load(file)

    if len(metadata) != len(cropped_region_coordinates):
        sys.exit("Metadata and cropped region coordinates do not match.")

    common_part = {
        'aachen': 2, 'bochum': 1, 'bremen': 2, 'cologne': 2, 'darmstadt': 2,
        'dusseldorf': 2, 'erfurt': 2, 'hamburg': 1, 'hanover': 1, 'jena': 2,
        'krefeld': 1, 'monchengladbach': 1, 'strasbourg': 1, 'stuttgart': 2,
        'tubingen': 2, 'ulm': 2, 'weimar': 2, 'zurich': 2
    }

    cropped_region_coordinates_dict = []

    for i in tqdm(range(len(metadata))): 
                
        original_image_path = metadata[i][str(i)]["rgb_full"]
        cropped_instance_rgb_path = metadata[i][str(i)]["cropped_instance_rgb"]
        bounding_box = cropped_region_coordinates[i]["bbox_coordinate"]
        seperated_anot_path = metadata[i][str(i)]["separated_anot"]
        anot_full_path = metadata[i][str(i)]["anot_full"]
        city = original_image_path.split('/')[-2]
        prefix_name = seperated_anot_path.split('/')[-1].replace('_anot.png', '')
        
        # For croped image  
        leftImg8bit_root = '/'.join(original_image_path.split('/')[1:-1]).replace(original_image_path.split('/')[-5], output_folder_name)      
        leftImg8bit_dir = leftImg8bit_root.replace(city, city + "_syn")
        if not os.path.exists(leftImg8bit_dir):
            os.makedirs(leftImg8bit_dir)
            print(f'Create dir {leftImg8bit_dir}')
        
        inpainted_dir = leftImg8bit_root.replace(city, city + '_syn').replace('leftImg8bit', 'gtFine') 
        if not os.path.exists(inpainted_dir):
            os.makedirs(inpainted_dir)
            print(f'Create dir {inpainted_dir}')
            
        # For uncropped image
        leftImg8bit_panorama_dir = leftImg8bit_root.replace('leftImg8bit', "leftImg8bit_panorama").replace(city, city + "_panorama")
        if not os.path.exists(leftImg8bit_panorama_dir):
            os.makedirs(leftImg8bit_panorama_dir)
            print(f'Create dir {leftImg8bit_panorama_dir}')
            

        for j in range(1, 3):  # Iterate over 1, 2, 3 for 000001, 000002, 000003
            inpainting_image_name = os.path.basename(seperated_anot_path).replace('anot', f'{model_type}_{j:06d}_{ver}')
            inpainting_image_path = os.path.join(inpainting_dir, 'train', city , inpainting_image_name)
            
            if os.path.exists(inpainting_image_path):
                inpainting_mask_image_path = seperated_anot_path.replace('anot', 'gtFine_instanceIds')
                inpainted_merge_img, normalized_inpainted_merge_img = inpaint_merge(
                    original_image_path, cropped_instance_rgb_path, inpainting_image_path,
                    inpainting_mask_image_path, bounding_box
                )
                crop_and_resize_images(inpainted_merge_img, normalized_inpainted_merge_img, seperated_anot_path, anot_full_path)

    # os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(os.path.join(output_folder_name, f'cropped_region_coordinates.json'), 'w') as f:
        json.dump(cropped_region_coordinates_dict, f, indent=4)
