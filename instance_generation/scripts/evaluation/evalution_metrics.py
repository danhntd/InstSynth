import PIL.Image as Image
from skimage.metrics import structural_similarity as ssim
import os
import argparse
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import clip
from skimage.color import rgb2gray
import torch
import torch.nn as nn
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from torch.utils.data import Dataset, DataLoader
import cv2
from pytorch_fid import fid_score

device = "cuda" if torch.cuda.is_available() else "cpu"

class SynDataset(Dataset):
    def __init__(self, groundtruth_path, generated_image_path, transform=None):
        self.gt_path = groundtruth_path
        self.gen_path = generated_image_path
        self.transform = transform
        self.gt_images = load_images(self.gt_path, transform=self.transform, read_image=True)
        self.gen_images = load_images(self.gen_path, transform=self.transform, read_image=True)

    def __len__(self):
        return len(self.gt_images)
    
    def __getitem__(self, idx):
        return self.gt_images[idx], self.gen_images[idx]
    
    def get_images(self):
        return self.gt_images, self.gen_images

def load_images(city_folder, transform=None, sample_id="recons", read_image=False):
    images = []
    
    if isinstance(city_folder, str):
        city_list = sorted(os.listdir(city_folder))

        city_list = [city for city in city_list if ("_" in city and len(city_list) > 18) or (len(city_list)==18)]
        for city in tqdm(city_list):
            image_folder = os.path.join(city_folder, city)
            image_list = sorted(os.listdir(image_folder))
            # print(city, ": ", len(image_list))
            for image_name in image_list:
                if sample_id not in image_name:
                    image_path = os.path.join(image_folder, image_name)
                    if read_image:
                        image = Image.open(image_path)
                        if transform:
                            image = transform(image)
                        images.append(image)
                    else:
                        images.append(image_path)
    if isinstance(city_folder, list):
        if read_image:
            for image_path in tqdm(city_folder, total=len(city_folder)):
                # breakpoint()
                image = Image.open(image_path)
                if transform:
                    image = transform(image)
                images.append(image)
        else:
            return city_folder
    # breakpoint()
    return images

def filtered_image_path(gt_path, gen_path, sample_id=None):
    filtered_gt_paths = []
    filtered_gen_paths = []
    gt_image_paths = load_images(gt_path, sample_id=sample_id)
    gen_image_paths = load_images(gen_path, sample_id=sample_id)
    # print("length of non-filtered gen path: ", len(gen_image_paths))
    for gen_image_path in gen_image_paths:
        if sample_id not in gen_image_path:
            gen_img_name = gen_image_path.split("/")[-1]
            splitted_name_image_id = gen_img_name.split("_")[:3]
            city_name = splitted_name_image_id[0]
            name_image_id = os.path.join(gt_path, city_name+"_syn", "_".join(splitted_name_image_id) + "_leftImg8bit.png")
            # breakpoint()
            idx = gt_image_paths.index(name_image_id)
            gt_image_path = gt_image_paths[idx]
            filtered_gt_paths.append(gt_image_path)
            filtered_gen_paths.append(gen_image_path)
    print("length of filtered gt path: ", len(filtered_gt_paths))
    print("length of filtered generated path: ", len(filtered_gen_paths))
    return filtered_gt_paths, filtered_gen_paths

def calc_ssim(image1_path, image2_path):
    # Load images
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Ensure images have the same size
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # Compute SSIM
    similarity_index = ssim(img1, img2)
    # print("similarity_index: ", similarity_index)
    return similarity_index

def psnr(img1, img2):
    # Read images
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    # Convert images to float32
    img1 = img1.astype('float32')
    img2 = img2.astype('float32')

    # Calculate MSE (Mean Squared Error)
    mse = np.mean((img1 - img2) ** 2)

    # If MSE is zero, PSNR is infinity
    if mse == 0:
        return float('inf')

    # Calculate PSNR
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def calc_ssim_psnr(gt_image_paths, gen_image_paths):

    # Initialize total SSIM value
    total_ssim = []
    total_psnr = []
    # Iterate through each pair of images
    for gt_image_path, gen_image_path in tqdm(zip(gt_image_paths, gen_image_paths), total=len(gt_image_paths)):
        # Calculate SSIM for the pair of images
        similarity_index = calc_ssim(gt_image_path, gen_image_path)
        psnr_value = psnr(gt_image_path, gen_image_path)
        total_psnr.append(psnr_value)
        total_ssim.append(similarity_index)
    print("total_ssim length: ", len(total_ssim))
    # Calculate average SSIM
    avg_ssim = np.mean(total_ssim)
    avg_psnr = np.mean(total_psnr)
    return avg_ssim, avg_psnr

def calc_clip(dataset):
    cos = nn.CosineSimilarity(dim=1)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    clip_score = []
    with torch.no_grad():
        for gt, gen in tqdm(dataset, total=len(dataset)):
            # breakpoint()
            gt = preprocess(gt).to(device)
            gen = preprocess(gen).to(device)
            gt_features = clip_model.encode_image(gt)
            gen_features = clip_model.encode_image(gen)
            cos_score = cos(gt_features, gen_features)
            cos_score_mean = cos_score.mean().item()
            clip_score.append(cos_score_mean)
    avg_clip = np.mean(clip_score)
    # print("CLIP score: ", avg_clip)
    return avg_clip

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth", required=True, type=str)
    parser.add_argument("--generated", required=True, type=str)
    parser.add_argument("--sample_id", type=str, default="000002")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    groundtruth_path = args.groundtruth
    generated_image_path = args.generated
    sample_id = args.sample_id
    # calculate CLIP Score
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
    ])
    
    filtered_gt_path, filtered_gen_path = filtered_image_path(groundtruth_path, generated_image_path, sample_id) 
    
    fid = fid_score.calc_fid([filtered_gt_path, filtered_gen_path], num_workers=1, batch_size=50, dims=2048)
    avg_ssim, avg_psnr = calc_ssim_psnr(filtered_gt_path, filtered_gen_path)
    dataset = SynDataset(filtered_gt_path, filtered_gen_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    avg_clip = calc_clip(dataloader)
    
    
    print("CLIP Score: ", avg_clip)
    print("PSNR: ", avg_psnr)
    print("SSIM: ", avg_ssim)
    print("FID: ", fid)
     