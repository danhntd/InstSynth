import argparse
import os
import random
from functools import partial
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np
import sys
# Adjust the Python path to include the parent directory of the `instance_generation` package
sys.path.append(str(Path(__file__).resolve().parent.parent))

from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPProcessor, CLIPModel
from models.model import BlendedLatentDiffusionSDXL
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import json
from utils import is_exist, get_metadata, get_cities, batch_to_device, get_clip_feature, get_masks, complete_mask, mask2box
import torchvision.transforms.functional as F
device = "cuda"

prompts = {
        "bus": ["a red bus", "a green bus"],
        "bicycle": ["a white bicycle", "a black bicycle"],
        "car": ["a red car", "a white car"],
        "motorcycle": ["a black motorcycle", "a white motorcycle"],
        "person": ["a woman", "a man"],
        "trailer": ["a green trailer", "a red trailer"],
        "train": ["a red train", "a black train"],
        "rider": ["a male rider", "a female rider"],
        "truck": ["a red truck", "a green truck"],
        "caravan": ["a white caravan", "a black caravan"]
    }

def generate_blended_samples(pipe, metadata, save_folder_name, city, args):
    metadata_path = os.path.join(args.folder, save_folder_name, f"metadata_output_blendeddiff_{city}.json")
    output_meta = {}
    seed = 1
    generator = torch.Generator(device=device).manual_seed(seed)
    
    for id, data in tqdm(metadata.items()):
        mask_path = data['cropped_instance_mask']
        input_image_path = data['cropped_instance_rgb']
        phrases = data['instance_class']
        state = input_image_path.split("/")[-3]
        output_folder = os.path.join(args.folder, save_folder_name, state, city)
        os.makedirs(output_folder, exist_ok=True)
        
        if not is_exist(input_image_path, output_folder):
            instance = random.choice(prompts[phrases])
            prompt = f"a photo of {instance}"
            input_name = "_".join(input_image_path.split("/")[-1].split(".")[0].split("_")[:3] + ["BlendedDiff"])
            output_meta[id] = {"prompt": prompt}
            outputs = pipe.edit_image(prompt=[prompt] * args.batch_size, init_image=input_image_path, mask=mask_path, generator=generator)
            
            for i, output in enumerate(outputs):
                img_name = f"{input_name}_{i+1:06d}_v1.png"
                output.save(os.path.join(output_folder, img_name))
                output_meta[id][f"sample_{i+1}"] = os.path.join(output_folder, img_name)
    
    with open(metadata_path, "w") as f:
        json.dump(output_meta, f, indent=4)

# GLIGEN Functions
def load_ckpt(ckpt_path):
    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    model.load_state_dict(saved_ckpt['model'])
    autoencoder.load_state_dict(saved_ckpt["autoencoder"])
    text_encoder.load_state_dict(saved_ckpt["text_encoder"])
    diffusion.load_state_dict(saved_ckpt["diffusion"])

    return model, autoencoder, text_encoder, diffusion, config

def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if isinstance(module, (GatedCrossAttentionDense, GatedSelfAttentionDense)):
            module.scale = alpha_scale

def alpha_generator(length, type=None):
    if type is None:
        type = [1, 0, 0]

    assert len(type) == 3 
    assert sum(type) == 1
    
    stage0_length = int(type[0] * length)
    stage1_length = int(type[1] * length)
    stage2_length = length - stage0_length - stage1_length
    
    decay_alphas = np.arange(start=0, stop=1, step=1 / stage1_length)[::-1] if stage1_length != 0 else []
    alphas = [1] * stage0_length + list(decay_alphas) + [0] * stage2_length
    
    assert len(alphas) == length
    return alphas

@torch.no_grad()
def prepare_batch(meta, batch=1, max_objs=30):
    phrases, images = meta.get("phrases"), meta.get("images")
    images = [None] * len(phrases) if images is None else images 
    phrases = [None] * len(images) if phrases is None else phrases 

    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).cuda()
    processor = CLIPProcessor.from_pretrained(version)

    boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs)
    text_masks = torch.zeros(max_objs)
    image_masks = torch.zeros(max_objs)
    text_embeddings = torch.zeros(max_objs, 768)
    image_embeddings = torch.zeros(max_objs, 768)
    
    text_features = [get_clip_feature(model, processor, phrase, is_image=False) for phrase in phrases]
    image_features = [get_clip_feature(model, processor, image, is_image=True) for image in images]

    for idx, (box, text_feature, image_feature) in enumerate(zip(meta['locations'], text_features, image_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        if text_feature is not None:
            text_embeddings[idx] = text_feature
            text_masks[idx] = 1 
        if image_feature is not None:
            image_embeddings[idx] = image_feature
            image_masks[idx] = 1 

    out = {
        "boxes": boxes.unsqueeze(0).repeat(batch, 1, 1),
        "masks": masks.unsqueeze(0).repeat(batch, 1),
        "text_masks": text_masks.unsqueeze(0).repeat(batch, 1) * complete_mask(meta.get("text_mask"), max_objs),
        "image_masks": image_masks.unsqueeze(0).repeat(batch, 1) * complete_mask(meta.get("image_mask"), max_objs),
        "text_embeddings": text_embeddings.unsqueeze(0).repeat(batch, 1, 1),
        "image_embeddings": image_embeddings.unsqueeze(0).repeat(batch, 1, 1)
    }

    return batch_to_device(out, device) 

@torch.no_grad()
def generate_gligen_samples(meta, metadata, save_folder_name, config, city, args, starting_noise=None): 
    model, autoencoder, text_encoder, diffusion, config, grounding_downsampler_input, grounding_tokenizer_input = meta.values()
    output_meta = {}
    
    for id, data in tqdm(metadata.items()):
        mask_path = data['cropped_instance_mask']
        input_image_path = data['cropped_instance_rgb']
        phrases = data['instance_class']
        state = input_image_path.split("/")[-3]
        output_folder = os.path.join(args.folder, save_folder_name, state, city)
        os.makedirs(output_folder, exist_ok=True)
        
        if not is_exist(input_image_path, output_folder, "GLIGEN"):
            instance = random.choice(prompts[phrases])
            prompt = f"a photo of {instance}"
            data = {
                "input_image": input_image_path,
                "mask": mask_path,
                "prompt": prompt,
                "phrases": [phrases],
                "locations": mask2box(mask_path)
            }
            batch = prepare_batch(data, config['batch_size'])

            context = text_encoder.encode([prompt] * config['batch_size'])
            uc = text_encoder.encode(config['batch_size'] * [""])
            if args.negative_prompt is not None:
                uc = text_encoder.encode(config['batch_size'] * [args.negative_prompt])

            alpha_generator_func = partial(alpha_generator, type=None)
            sampler_class = DDIMSampler if config['no_plms'] else PLMSSampler
            sampler = sampler_class(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
            steps = 250 if config['no_plms'] else 50

            inpainting_mask = get_masks(mask_path, config['batch_size'])
            input_image = F.pil_to_tensor(Image.open(input_image_path).convert("RGB").resize((512, 512)))
            input_image = (input_image.float().unsqueeze(0).cuda() / 255 - 0.5) / 0.5
            z0 = autoencoder.encode(input_image)
            masked_z = z0 * inpainting_mask
            inpainting_extra_input = torch.cat([masked_z, inpainting_mask], dim=1)

            grounding_input = grounding_tokenizer_input.prepare(batch)
            grounding_extra_input = None
            if grounding_downsampler_input is not None:
                grounding_extra_input = grounding_downsampler_input.prepare(batch)

            input = {
                "x": starting_noise, 
                "timesteps": None, 
                "context": context, 
                "grounding_input": grounding_input,
                "inpainting_extra_input": inpainting_extra_input,
                "grounding_extra_input": grounding_extra_input,
            }

            shape = (config['batch_size'], model.in_channels, model.image_size, model.image_size)
            samples_fake = sampler.sample(S=steps, shape=shape, input=input, uc=uc, guidance_scale=config['guidance_scale'], mask=inpainting_mask, x0=z0)
            samples_fake = autoencoder.decode(samples_fake)

            start = 0
            image_ids = list(range(start, start + 2))
            input_name = "_".join(input_image_path.split("/")[-1].split(".")[0].split("_")[:3] + ["GLIGEN"])
            output_meta[id] = {"prompt": prompt}
            for image_id, sample in zip(image_ids, samples_fake):
                img_name = f"{input_name}_{image_id + 1:06d}_v1.png"
                sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
                sample = sample.cpu().numpy().transpose(1, 2, 0) * 255
                sample = Image.fromarray(sample.astype(np.uint8))
                sample.save(os.path.join(output_folder, img_name))
                output_meta[id][f"sample_{image_id + 1}"] = os.path.join(output_folder, img_name)
    with open(os.path.join(args.folder, save_folder_name, f"metadata_output_{city}.json"), "w") as fOut:
        json.dump(output_meta, fOut, indent=4)

# DiffPainting Functions
def generate_diffpainting_samples(pipe, metadata, save_folder_name, city, args):
    output_meta = {}
    seed = 1
    generator = torch.Generator(device=device).manual_seed(seed)

    for id, data in tqdm(metadata.items()):
        mask_path = data['cropped_instance_mask']
        input_image_path = data['cropped_instance_rgb']
        phrases = data['instance_class']
        state = input_image_path.split("/")[-3]
        output_folder = os.path.join(args.folder, save_folder_name, state, city)
        os.makedirs(output_folder, exist_ok=True)
        if not is_exist(input_image_path, output_folder, "DiffPainting"):
            image = Image.open(input_image_path)
            mask_image = Image.open(mask_path)
            instance = random.choice(prompts[phrases])
            prompt = "a photo of " + instance
            input_name = "_".join(input_image_path.split("/")[-1].split(".")[0].split("_")[:3] + ["DiffPainting"])
            output_meta[id] = {}
            output_meta[id]["prompt"] = prompt
            outputs = pipe(prompt=prompt, image=image, mask_image=mask_image, num_images_per_prompt=args.batch_size, generator=generator).images
            
            for i, output in enumerate(outputs):
                img_name = f"{input_name}_{i+1:06d}_v1.png"
                output.save(os.path.join(output_folder, img_name))
                output_meta[id][f"sample_{i+1}"] = os.path.join(output_folder, img_name)
    
    with open(os.path.join(args.folder, save_folder_name, f"metadata_output_{city}.json"), "w") as fOut:
        json.dump(output_meta, fOut, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["bld", "gligen", "diffpainting"], required=True, help="Select the mode to run: 'bld', 'gligen', or 'diffpainting'")
    
    # Common arguments
    parser.add_argument("--folder", type=str, default="generation_samples", help="root folder for output")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--device", type=str, default="cuda")

    # Blended Diffusion arguments
    parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="The path to the HuggingFace model")
    parser.add_argument("--blending_start_percentage", type=float, default=0.25, help="The diffusion steps percentage to jump")

    # GLIGEN arguments
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="guidance scale")
    parser.add_argument("--negative_prompt", type=str, default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help="negative prompt")
    
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "bld":
        blended_save_folder_name = "BlendedDiff"
        bld = BlendedLatentDiffusionSDXL.from_pretrained(args.model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        bld.args = args
        bld.to(args.device)

        with open(os.path.join(args.folder, "metadata_refined.json"), "r") as fIn:
            metadata = json.load(fIn)

        cities = get_cities(os.path.join(args.folder, "leftImg8bit/train/"))
        cities.sort()

        filtered_metadata = {}
        filtered_metadata = {city: {} for city in cities}

        for id, data in metadata.items():
            city = data["cropped_instance_rgb"].split("/")[-1].split("_")[0]
            if city not in filtered_metadata:
                filtered_metadata[city] = {}
            filtered_metadata[city][id] = data
        
        for city in cities:
            print(f"Current city: {city}")
            generate_blended_samples(bld, filtered_metadata[city], blended_save_folder_name, city, args)


    elif args.mode == "gligen":
        import sys
        sys.path.append("GLIGEN/")
        ckpt = "GLIGEN/gligen_checkpoints/checkpoint_inpainting_text.pth"
        gligen_save_folder_name = "gligen"
        starting_noise = None

        model, autoencoder, text_encoder, diffusion, config = load_ckpt(ckpt)
        print(config.keys())
        grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
        model.grounding_tokenizer_input = grounding_tokenizer_input

        grounding_downsampler_input = None
        if "grounding_downsampler_input" in config:
            grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])

        config.update(vars(args))
        config = OmegaConf.create(config)

        meta = {
            "model": model,
            "autoencoder": autoencoder,
            "text_encoder": text_encoder,
            "diffusion": diffusion,
            "config": config,
            "grounding_downsampler_input": grounding_downsampler_input,
            "grounding_tokenizer_input": grounding_tokenizer_input
        }

        with open(os.path.join(args.folder, "metadata_refined.json"), "r") as fIn:           
            metadata = json.load(fIn)

        cities = get_cities(os.path.join(args.folder, "leftImg8bit/train/"))
        cities.sort()

        filtered_metadata = {}
        filtered_metadata = {city: {} for city in cities}

        for id, data in metadata.items():
            city = data["cropped_instance_rgb"].split("/")[-1].split("_")[0]
            if city not in filtered_metadata:
                filtered_metadata[city] = {}
            filtered_metadata[city][id] = data
        
        for city in cities:
            print(f"Current city: {city}")
            generate_gligen_samples(meta, filtered_metadata[city], gligen_save_folder_name, config, city, args, starting_noise)

    elif args.mode == "diffpainting":
        DiffInpainting_save_folder_name = "DiffPainting"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
        )

        pipe.to(args.device)
        pipe.enable_attention_slicing()
        with open(os.path.join(args.folder, "metadata_refined.json"), "r") as fIn:
            metadata = json.load(fIn)
        
        cities = get_cities(os.path.join(args.folder, "leftImg8bit/train/"))
        cities.sort()

        filtered_metadata = {}
        filtered_metadata = {city: {} for city in cities}

        for id, data in metadata.items():
            city = data["cropped_instance_rgb"].split("/")[-1].split("_")[0]
            if city not in filtered_metadata:
                filtered_metadata[city] = {}
            filtered_metadata[city][id] = data

        for city in cities:
            print(f"Current city: {city}")
            generate_diffpainting_samples(pipe, filtered_metadata[city], DiffInpainting_save_folder_name, city, args)
