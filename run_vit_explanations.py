from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from samples.CLS2IDX import CLS2IDX
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_explanation_generator import LRP
import torchvision.transforms as T
import torchvision 
from dataset import IMAGENET_STD, IMAGENET_MEAN, denormalize

import os 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--midx", type=int, required=True)
args = parser.parse_args()

save_dir = "results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def post_process_attribution(transformer_attribution):
    transformer_attribution = transformer_attribution.detach().cpu() 
    if transformer_attribution.shape[-1] ==196:
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    return transformer_attribution

data_path = '/data/ImageNet1k'

transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(), 
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])

valid_dataset = torchvision.datasets.ImageNet(root=data_path, split="val", transform=transform)

# initialize ViT pretrained
model = vit_LRP(pretrained=True).cuda()
model.eval()
attribution_generator = LRP(model)

from tqdm import tqdm 
methods = ['transformer_attribution', 'rollout', 'full', 'last_layer', 'last_layer_attn', 'second_layer', ]

method = methods[args.midx]

attrs = np.zeros(shape=(len(valid_dataset), 224,224)).astype(np.float16)
pbar = tqdm(range(len(valid_dataset)))
pbar.set_description(f"[👾] {args.midx}: {method} |")
for idx in pbar:
    original_image, class_index = valid_dataset[idx]
    original_image = original_image.unsqueeze(0).cuda()
    transformer_attribution = attribution_generator.generate_LRP(original_image, method=method, index=class_index)
    transformer_attribution = post_process_attribution(transformer_attribution)
    attrs[idx] = transformer_attribution.astype(np.float16)
    break
    
np.save(os.path.join(save_dir, method+".npy"), attrs)