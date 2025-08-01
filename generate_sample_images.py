import os
import random
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as TF
from PIL import Image
import json
import numpy as np

from model import CCAE  # Replace with your actual model class
from data_loader import ImageDataset     # Ensure this matches the location of your class

# ---------------------------
# Utility: Save comparison plot
# ---------------------------
def save_comparison(input_img, gt_img, pred_img, idx, output_dir):
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    
    axs[0].imshow(input_img.squeeze(), cmap='gray')
    axs[0].set_title('Input')
    axs[0].axis('off')
    
    axs[1].imshow(gt_img.squeeze(), cmap='gray')
    axs[1].set_title('Ground Truth')
    axs[1].axis('off')

    axs[2].imshow(pred_img.squeeze(), cmap='gray')
    axs[2].set_title('Reconstruction')
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sample_{idx}.png'))
    plt.close()

# ---------------------------
# Main script
# ---------------------------
def main(args):
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    # Load normalization stats
    with open(args.norm_file, "r") as f:
        norm_stats = json.load(f)
    means = np.array(norm_stats["means"])
    stds = np.array(norm_stats["stds"])

    # Load data
    df = pd.read_csv(args.csv_file)
    total_samples = len(df)
    indices = random.sample(range(total_samples), args.num_samples)

    # Dataset with correct normalization
    dataset = ImageDataset(df, args.image_folder, means, stds)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    # Load model
    model = CCAE(base_channels=32).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    # Run inference
    with torch.no_grad():
        for idx, (img_in, img_out, cond) in enumerate(loader):
            img_in, cond = img_in.to(device), cond.to(device)
            pred, global_step = model(img_in, cond)

            save_comparison(img_in.cpu().numpy(),img_out.cpu().numpy(), pred.cpu().numpy(), idx, args.output_dir)

    print(f"Saved {args.num_samples} sample comparisons to '{args.output_dir}'.")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='best_model/best_ema_model.pth', help='Path to trained model checkpoint')
    parser.add_argument('--csv_file', type=str, default='../VD-FEBE-Data/PreIP_512_FEB25/data_labels.csv', help='Path to CSV file')
    parser.add_argument('--image_folder', type=str, default='../VD-FEBE-Data/PreIP_512_FEB25/XYImages_Smoothed', help='Path to image folder')
    parser.add_argument('--norm_file', type=str, default='../VD-FEBE-Data/PreIP_512_FEB25/normalization.json', help='Path to normalization stats JSON')
    parser.add_argument('--output_dir', type=str, default='samples', help='Directory to save output comparisons')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of random samples to visualize')
    args = parser.parse_args()

    main(args)
