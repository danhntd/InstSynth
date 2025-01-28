import sys
import argparse
import json
import os
import cv2
import numpy as np
from PIL import Image 

from utils.instance_class import *
from utils.labels import *

def set_values_to_zero(image_array, coordinates_list):
    """
    Set values of the 'image_array' at specified coordinates to 0.
    
    Args:
        image_array (numpy.ndarray): 2D array containing image values.
        coordinates_list (list): List of coordinates (tuples) to set values to 0.
    
    Returns:
        numpy.ndarray: Modified 2D array after setting values at specified coordinates to 0.
    """
    # Make a copy of the image array to avoid modifying the original array
    modified_array = image_array.copy()
    
    # Iterate through the list of coordinates and set values at those coordinates to 0
    for coord in coordinates_list:
        y, x = coord  # Get coordinates (y, x)
        if 0 <= y < image_array.shape[0] and 0 <= x < image_array.shape[1]:
            modified_array[y, x] = 0
    
    return modified_array


def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4
    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy

def instances2dict_with_polygons(imageFileName, changed_coordinated_dict, verbose=False):
    # Load image
    img = Image.open(imageFileName)
    imgNp = np.array(img)

    # Initialize categories contain polygon
    instances = {}
    instances['imgWidth'] = img.size[0]
    instances['imgHeight'] = img.size[1]
    instances['objects'] = []
    
    # Loop through all instance ids in instance image   
    for instanceId in np.unique(imgNp):

        if instanceId < 24000:
            continue
        instanceObj = Instance(imgNp, instanceId)

        # Create objects contain object label and polygon
        objects = {}
        objects['label'] = id2label[instanceObj.labelID].name
        
        # Initialize instances['objects'] as an empty list before the loop
        if id2label[instanceObj.labelID].hasInstances:
            mask = (imgNp == instanceId).astype(np.uint8)
            contour, hier = findContours(mask.copy(), cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
            
            list_polygons = [point.flatten().tolist() for cont in contour for point in cont]
            objects['polygon'] = list_polygons
            
            if len(list_polygons) < 4:

                # Get coordinates for change the value of instance area to 0 
                coordinates_list = np.argwhere(mask)

                # For color image 
                color_image_path = imageFileName.replace('instanceIds','color')
                colorNp = np.array(Image.open(color_image_path))
                new_color_img = set_values_to_zero(colorNp, coordinates_list)
                colorNp = new_color_img
                Image.fromarray(colorNp).save(color_image_path)

                # For labelIds image 
                labelIds_image_path = imageFileName.replace('instanceIds','labelIds')
                labelIdsNp = np.array(Image.open(labelIds_image_path))
                new_labelIds_img = set_values_to_zero(labelIdsNp, coordinates_list)
                labelIdsNp = new_labelIds_img
                Image.fromarray(labelIdsNp).save(labelIds_image_path)
                
                # For InstanceIds image 
                new_instanceIds_img = set_values_to_zero(imgNp, coordinates_list)
                imgNp = new_instanceIds_img
                Image.fromarray(imgNp).save(imageFileName)

                # Save the coordinates of the cropped region
                entry = {
                        'cropped_instances_mask': str(imageFileName),
                        'bbox_coordinate': coordinates_list.tolist()
                }
                changed_coordinated_dict.append(entry)
                
            else:    
                instances['objects'].append(objects)

    return instances


def convert_cityscapes_instance_only(data_dir):
    """Convert from cityscapes format to COCO instance seg format - polygons"""
    sets = [
        'leftImg8bit/train',
        'leftImg8bit/val'
    ]

    ann_dirs = [
        'gtFine/train',
        'gtFine/val',
    ]
    
    
    image_folders = [file for file in sorted(os.listdir( os.path.join(data_dir, ann_dirs[0] ))) if "_syn" in file]
    
    # Dict to save the coordinates were change if less than 4 polygons
    changed_coordinated_dict = []

    for image_folder in image_folders[:]:
        image_files = [file for file in sorted(os.listdir( os.path.join(data_dir, ann_dirs[0], image_folder ))) if "gtFine_instanceIds" in file]
        
        for mask_img in image_files[:]:
            fullname = os.path.join(data_dir, ann_dirs[0], image_folder, mask_img)
            instance_dict = instances2dict_with_polygons(fullname, changed_coordinated_dict, verbose=False)
            # Save file contain polygons
            with open(fullname.replace("instanceIds", "polygons").replace('png','json'), 'w') as outfile:
                outfile.write(json.dumps(instance_dict, indent=4))        

    # Save the coordinates of the cropped region.
    name = f'changed_coordinated_instance_less_than_4_polygons.json'
    with open(os.path.join(data_dir, name), 'w') as f:
      json.dump(changed_coordinated_dict, f, indent=4)    


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument('--dataset', help="cityscapes", default='cityscapes', type=str)
    parser.add_argument('--datadir', help="data dir for annotations to be converted", default="data/cityscapes", type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "cityscapes":
        convert_cityscapes_instance_only(args.datadir)
    else:
        print("Dataset not supported: %s" % args.dataset)
