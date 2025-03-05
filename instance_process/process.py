import os
import numpy as np
from PIL import Image
import json
import argparse
import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process Cityscapes dataset to extract and crop instances.")
    parser.add_argument('--train_folder', type=str, default='../cityscapes_synthesis/', required=True, help='Path to the train folder containing Cityscapes data.')
    parser.add_argument('--k', type=int, default=3, help='Number of largest instances to extract from each mask image.')
    return parser.parse_args()


def create_anot_topk(mask_img_name, k):
    """
    Creates separate annotation masks for the top-k largest instances in the given mask image.
    Each instance is extracted based on its unique pixel value and saved as a separate mask image.
    
    Args:
        mask_img_name (str): Name of the mask image file.
        k (int): Number of largest instances to extract.
    """
    
    mask_img_path = os.path.join(train_folder_path, image_folder, mask_img_name)
    img_arr = np.array(Image.open(mask_img_path))
    unique_values, counts = np.unique(img_arr, return_counts=True)
    pairs = [[value, count] for value, count in zip(unique_values, counts)]
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    # Filter top values based on threshold k
    filtered_sorted = [item for item in sorted_pairs if item[0] >= 24000]
    top_values = filtered_sorted[:k] if len(filtered_sorted) >= k else filtered_sorted

    for value in top_values:
        img_temp = img_arr.copy()
        img_temp[img_temp != value[0]] = 0

        # Create a new name for annotation image
        parts = mask_img_name.split('_')
        parts[common_part[image_folder]] = f"{str(value[0])[:2]}{str(value[0])[-2:]}{mask_img_name.split('_')[common_part[image_folder]][-2:]}"
        new_mask_img_name = '_'.join(parts)
        separated_anot_name = new_mask_img_name.replace('gtFine_instanceIds', 'anot')
        anot_path = os.path.join(anot_dir, separated_anot_name)
        Image.fromarray(img_temp).save(anot_path)

        global count_metadata
        matadata_dict.append({
            str(count_metadata): {
                "rgb_full": mask_img_path.replace('gtFine_instanceIds', 'leftImg8bit').replace('gtFine', 'leftImg8bit'),
                "anot_full": mask_img_path,
                "separated_anot": anot_path,
                "cropped_instance_mask": anot_path.replace('anot', 'gtFine_instanceIds'),
                "cropped_instance_rgb": anot_path.replace('anot', 'leftImg8bit').replace('gtFine', 'leftImg8bit'),
                "instance_class": labels[int(str(value[0])[:2])]
            }
        })
        count_metadata += 1


def crop_anot_images():
    """
    Crops anot rectangular regions around the extracted instances in the annotation masks.
    Ensures that the cropped regions form square bounding boxes, adjusting dimensions as needed.
    """
    print("Starting to crop annotation images for each extracted instance...")
    for i, data in tqdm.tqdm(enumerate(matadata_dict), total=len(matadata_dict), desc="Cropping Anot Images"):
        input_path = data[str(i)]["separated_anot"]
        output_path_anot = data[str(i)]["cropped_instance_mask"]

        image = Image.open(input_path)
        image_array = np.array(image)
        image_array[image_array != 0] = 255
        seperated_anot = Image.fromarray(image_array.astype(np.uint8))
        non_zero_indices = seperated_anot.point(lambda p: p != 0)
        bounding_box = non_zero_indices.getbbox()

        if not bounding_box:
            continue

        min_x, min_y, max_x, max_y = bounding_box
        bb_width, bb_height = max_x - min_x, max_y - min_y

        # Calculate square bounding box dimensions
        if bb_width >= bb_height:
            min_y_target = max(0, min_y - (bb_width - bb_height) // 2)
            max_y_target = min_y_target + bb_width
            min_x_target, max_x_target = min_x, max_x
        else:
            min_x_target = max(0, min_x - (bb_height - bb_width) // 2)
            max_x_target = min_x_target + bb_height
            min_y_target, max_y_target = min_y, max_y

        # Crop mask image
        cropped_image = image.crop((min_x_target, min_y_target, max_x_target, max_y_target))
        cropped_array = np.array(cropped_image, dtype='uint8')
        cropped_array[cropped_array != 0] = 255
        Image.fromarray(cropped_array).save(output_path_anot)

        cropped_region_coordinates_rectangle_dict.append({
            'cropped_instance_mask': output_path_anot,
            'bbox_coordinate': (min_x_target, min_y_target, max_x_target, max_y_target)
        })


def crop_leftImg8bit_images():
    """
    Crops the corresponding RGB images for each extracted annotation instance using the bounding box coordinates.
    Ensures the RGB crops match the dimensions of the cropped annotation masks.
    """
    print("Starting to crop RGB images for each extracted annotation instance...")
    for i, data in tqdm.tqdm(enumerate(matadata_dict), total=len(matadata_dict), desc="Cropping RGB Images"):
        rgb_raw_path = data[str(i)]["rgb_full"]
        rgb_crop_path = data[str(i)]["cropped_instance_rgb"]
        min_x_target, min_y_target, max_x_target, max_y_target = cropped_region_coordinates_rectangle_dict[i]['bbox_coordinate']

        rgb_raw = Image.open(rgb_raw_path)
        rgb_crop = rgb_raw.crop((min_x_target, min_y_target, max_x_target, max_y_target))
        rgb_crop.save(rgb_crop_path)


if __name__ == "__main__":
    
    args = parse_arguments()
    train_folder = args.train_folder
    k = args.k
    train_folder_path = os.path.join(train_folder, 'gtFine/train/')

    image_folders = [folder for folder in sorted(os.listdir(train_folder_path)) if not folder.endswith(('_inpainted', '_syn'))]

    common_part = {'aachen': 2, 'bochum': 1, 'bremen': 2, 'cologne': 2, 'darmstadt': 2,
                   'dusseldorf': 2, 'erfurt': 2, 'hamburg': 1, 'hanover': 1, 'jena': 2, 
                   'krefeld': 1, 'monchengladbach': 1, 'strasbourg': 1, 'stuttgart': 2,
                   'tubingen': 2, 'ulm': 2, 'weimar': 2, 'zurich': 2}

    labels = {
        24: 'person', 25: 'rider', 26: 'car', 27: 'truck', 28: 'bus',
        29: 'caravan', 30: 'trailer', 31: 'train', 32: 'motorcycle', 33: 'bicycle'
    }

    cropped_region_coordinates_rectangle_dict = []
    matadata_dict = []
    count_metadata = 0

    # 1. Extract top-k largest instances from annotation masks.
    for image_folder in image_folders[:]:
        print("Processing City:", image_folder)
        mask_img_names = [file for file in sorted(os.listdir(os.path.join(train_folder_path, image_folder))) if "gtFine_instanceIds" in file]

        k = 3
        anot_dir = os.path.join(train_folder_path, image_folder + '_syn')
        os.makedirs(anot_dir, exist_ok=True)

        leftImg8bit_dir = os.path.join(train_folder_path.replace('gtFine', 'leftImg8bit'), image_folder + '_syn')
        os.makedirs(leftImg8bit_dir, exist_ok=True)

        for mask_img_name in  tqdm.tqdm(mask_img_names, desc=f"Processing Mask Images in {image_folder}: ", unit="image"):
            create_anot_topk(mask_img_name, k)

    # 2. Crop anot regions for these instances.
    crop_anot_images()
    
    # 3. Save metadata and bounding box information for further processing.
    crop_leftImg8bit_images()


    with open(os.path.join(train_folder, 'cropped_region_coordinates.json'), 'w') as f:
        json.dump(cropped_region_coordinates_rectangle_dict, f, indent=4)

    with open(os.path.join(train_folder, 'metadata.json'), 'w') as f:
        json.dump(matadata_dict, f, indent=4)