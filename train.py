import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_loader import get_dataloaders
from model import CCAE
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import imageio.v2 as imageio
import glob
from torch.amp import GradScaler, autocast # FOR AMP
from ema import EMA
import time
import numpy as np

# LPIPS
import lpips

# Hardcoded paths
CSV_FILE = "../VD-FEBE-Data/PreIP_512_FEB25/data_labels.csv"
#IMAGE_FOLDER = "../VD-FEBE-Data/PreIP_512_FEB25/XYImages_Filtered"
IMAGE_FOLDER = "../VD-FEBE-Data/PreIP_512_FEB25/XYImages_Smoothed"

# Configuration
device_name = "cuda:2" if torch.cuda.is_available() else "cpu"
# torch.cuda.set_per_process_memory_fraction(0.8, device=0)
device = torch.device(device_name)
torch.backends.cudnn.benchmark = True
batch_size = 32
num_epochs = 2000
early_stopping_patience = 50
image_size = 512
condition_dim = 4
learning_rate = 1e-4
lr_patience = np.round(early_stopping_patience/2)
# LPIPS
lpips_split_size = 4

# Create directories for saving models and outputs
vis_dir = "epoch_debug_images"
os.makedirs(vis_dir, exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("best_model", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Paths
checkpoint_path = "checkpoints/checkpoint.pth"
best_model_path = "best_model/best_model.pth"
best_ema_model_path = "best_model/best_ema_model.pth"
final_model_path = "output/final_model.pth"
final_ema_model_path = "output/final_model_ema.pth"

# === Custom Loss Functions ===
# Weighted MSE
class WeightedMSELoss(nn.Module):
    def __init__(self, epsilon=1e-3, foreground_weight=10.0):
        super().__init__()
        self.epsilon = epsilon
        self.foreground_weight = foreground_weight

    def forward(self, input, target):
        # Create weight mask: foreground pixels get a higher weight
        weight_1 = torch.where(target > self.epsilon, self.foreground_weight, 1.0)
        weight_2 = torch.where(input > self.epsilon, self.foreground_weight, 1.0)
        loss = (weight_1*weight_2 * (input - target) ** 2).mean()
        return loss

class FocalMSELoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        error = (input - target).pow(2)
        weights = error.detach().pow(self.gamma)
        loss = weights * error
        return loss.mean() if self.reduction == 'mean' else loss.sum()


# === Create dataloaders ===
train_loader, val_loader = get_dataloaders(batch_size=batch_size, num_workers=0, pin_memory=True, csv_file=CSV_FILE, image_folder=IMAGE_FOLDER)

# === Fixed validation sample ===
val_iter = iter(val_loader)
val_sample = next(val_iter)

# Initialize model, optimizer, and custom loss

# LPIPS
##########################################
# Initialize LPIPS model
lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

# Disable gradients for LPIPS net
for param in lpips_loss_fn.parameters():
    param.requires_grad = False

# Hyperparameters for blending
start_lpips_weight = 0.0
max_lpips_weight = 0.9   # you can adjust how strong LPIPS becomes
blend_warmup_steps = 10000  # number of steps over which we increase LPIPS weight
lpips_decay_patience = np.round(early_stopping_patience/2.5)  # Start decay after 20 stagnant epochs
lpips_decay_target_weight = 0.1  # Final reduced LPIPS weight
lpips_decay_started = False
##########################################
criterion = nn.MSELoss()#WeightedMSELoss(epsilon=1e-3, foreground_weight=50.0).to(device) # Define the weighted MSE loss
#criterion = FocalMSELoss(gamma=2.0).to(device) # Define the focal MSE loss
model = CCAE(base_channels=32).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_patience, verbose=True)
scaler = GradScaler(device=device_name) # FOR AMP
ema = EMA(model, beta=0.999)
ema_model = ema.ema_model


# Tracking losses
train_losses = []
val_losses = []
train_cos_sims = []
val_cos_sims = []
best_val_loss = float("inf")
patience_counter = 0
start_epoch = 0  # To track resumption

# Load checkpoint if available
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler.load_state_dict(checkpoint["scaler_state"]) # FOR AMP
    best_val_loss = checkpoint["best_val_loss"]
    patience_counter = checkpoint["patience_counter"]
    start_epoch = checkpoint["epoch"] + 1  # Resume from next epoch
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    ema_model.load_state_dict(checkpoint["ema_state"])
    #train_cos_sims = checkpoint.get("train_cos_sims", [])
    #val_cos_sims = checkpoint.get("val_cos_sims", [])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])
    print(f"Resuming training from epoch {start_epoch}")


        
# Debugging images
def save_debug_images(epoch, model, val_sample, device):
    # model.eval() # Using EMA so not needed
    with torch.no_grad():
        image_in, image_out, conditions = val_sample  # Assuming a single image and its conditioning vector
        if image_in.ndim == 3:
            image_in = image_in.unsqueeze(0)  # Make it (1, C, H, W)
        image_in = image_in.to(device)          # Shape: (1, 1, 512, 512)
        if image_out.ndim == 3:
            image_out = image_out.unsqueeze(0)  # Make it (1, C, H, W)
        image_out = image_out.to(device)          # Shape: (1, 1, 512, 512)
        if conditions.ndim == 1:
            conditions = conditions.unsqueeze(0)
        conditions = conditions.to(device) # Shape: (1, 4)
        
        reconstructed, global_step = model(image_in, conditions)        

        # Plot and save
        fig, axs = plt.subplots(1, 5, figsize=(18, 5))

        axs[0].imshow(image_in[0].squeeze().cpu(), cmap='gray')
        axs[0].set_title("Input - " + str(epoch))
        
        axs[1].imshow(image_out[0].squeeze().cpu(), cmap='gray')
        axs[1].set_title("GT - " + str(epoch))
        
        axs[2].imshow(reconstructed[0].squeeze().cpu(), cmap='gray')
        axs[2].set_title("Reconstructed - " + str(epoch))
        
        axs[3].imshow(	abs(image_out[0] - reconstructed[0]).squeeze().cpu(), cmap='hot')
        axs[3].set_title("Abs Diff - " + str(epoch))

        for ax in axs:
            ax.axis('off')

        fig.tight_layout()
        plt.savefig(f"{vis_dir}/epoch_{epoch:03}.png")
        plt.close()



def make_gif(image_dir, output_file="output/training_progress.gif"):
    image_paths = sorted(glob.glob(f"{image_dir}/epoch_*.png"))
    images = [imageio.imread(p) for p in image_paths]
    imageio.mimsave(output_file, images, duration=0.8)


# LPIPS
###############################################
def get_ema_decay(global_step, base=0.9, final=0.999, warmup_steps=10000):
    t = min(global_step / warmup_steps, 1.0)
    return base + (final - base) * t

def sigmoid_rampup(current, rampup_length):
    """Exponential sigmoid rampup from 0 to 1 over `rampup_length` steps."""
    if rampup_length == 0:
        return 1.0
    #current = float(current)
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

def get_loss_weights(step, max_lpips_weight = max_lpips_weight, warmup_steps=blend_warmup_steps, patience_counter = 0, prev_patience_counter = 0, lpips_decay_patience=lpips_decay_patience, lpips_decay_started=False):

    # LPIPS Ramp-up Phase
    step = float(step)
    #print(step)
    if step < warmup_steps:
        # LPIPS ramps from 0 to 0.9
        lpips_weight = max_lpips_weight*sigmoid_rampup(step,warmup_steps)
        #max_lpips_weight * progress

    # LPIPS Constant Phase
    elif not lpips_decay_started and patience_counter < lpips_decay_patience:
        lpips_weight = max_lpips_weight

    # Trigger LPIPS Decay Phase
    elif not lpips_decay_started and patience_counter >= lpips_decay_patience:
        lpips_decay_started = True
        decay_progress = 0.0
        lpips_weight = max_lpips_weight

    # LPIPS Decay Phase
    elif lpips_decay_started:
        if patience_counter > prev_patience_counter:
            decay_progress = min(
                (patience_counter - lpips_decay_patience) / (early_stopping_patience - lpips_decay_patience),
                1.0
            )
            lpips_weight = (
                max_lpips_weight * (1.0 - decay_progress)
                + lpips_decay_target_weight * decay_progress
            )
            
    # L1 weight 1 - LPIPS_weight
    l1_w = 1.0 - lpips_weight

    # Store patience counter
    prev_patience_counter = patience_counter
    
    return l1_w, lpips_weight
################################################

def train(model, train_loader, val_loader, optimizer, device, num_epochs, val_sample):
    global best_val_loss, patience_counter, start_epoch, train_losses, val_losses, prev_patience_counter, patience_counter
    patience_counter = 0
    prev_patience_counter = patience_counter
    for epoch in range(start_epoch, num_epochs):
        # For Weighted MSE
        #if epoch % 10 == 0:  # Every 10 epochs, lower the foreground weight
        #    criterion.foreground_weight *= 0.75
        start_time = time.time()
        model.train()
        #epoch_cos_sims = []
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]")

        accum_steps = 1#16/batch_size  # Simulates batch size of accum_steps*actual batch size
        optimizer.zero_grad(set_to_none=True)  # zero once per accumulation cycle
        #save_debug_images(epoch, ema_model, val_sample, device) # debugging
        for step, (images_in, images_out, conditions) in enumerate(loop):
            images_in, images_out, conditions = images_in.to(device), images_out.to(device), conditions.to(device)

            B = images_in.shape[0]
            
            with autocast(device_type=device_name): # FOR AMP
#                # input for self-conditioning
#                use_self_condition = torch.rand(1).item() < 0.5
#                if use_self_condition:
#                    with torch.no_grad():
#                        x0_prev = noisy_images + model(noisy_images, t, conditions).to(device)
#                else:
#                    x0_prev = None
                
                # Reconstruct image_out using model
                reconstruction, global_step = model(images_in, conditions)#, x0_prev)
        
                # Compute loss
                mse_loss = criterion(reconstruction, images_out)#F.mse_loss(reconstruction, images_out)
                # LPIPS
                #####################################################
                # LPIPS loss (safe microbatched + no_grad)
                lpips_parts = []
                batch_size = reconstruction.size(0)
                
                with torch.no_grad():
                    for i in range(0, batch_size, lpips_split_size):
                        recon_split = reconstruction[i:i+lpips_split_size]
                        target_split = images_out[i:i+lpips_split_size]
                        lpips_part = lpips_loss_fn(recon_split, target_split).mean()
                        lpips_parts.append(lpips_part)
                
                lpips_loss_value = torch.stack(lpips_parts).mean()

                # Compute current loss weight
                #print(global_step)
                mse_weight, lpips_weight = get_loss_weights(step=global_step, max_lpips_weight = max_lpips_weight, warmup_steps=blend_warmup_steps, patience_counter=patience_counter, prev_patience_counter=prev_patience_counter, lpips_decay_patience=lpips_decay_patience, lpips_decay_started=lpips_decay_started)
                #mse_weight, lpips_weight = get_loss_weights(global_step)#,blend_warmup_steps)
                
                loss = mse_weight * mse_loss + lpips_weight * lpips_loss_value
                
                ######################################################
                #loss = loss / accum_steps # Normalisation

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                # Backpropagation
                
                # loss.backward()
                # optimizer.step()
                ### FOR AMP''
                
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema_decay = get_ema_decay(global_step, base=0.9, final=0.999, warmup_steps=10000)
                ema.decay = ema_decay
                ema.update()
                model.step()

            train_loss += loss.item() * accum_steps
            #cos_sim = F.cosine_similarity(reconstruction.view(B, -1), images_out.view(B, -1), dim=1).detach().mean().item()

            #epoch_cos_sims.append(cos_sim)
            loop.set_postfix(loss=loss.item()*accum_steps)#, cos_sim=cos_sim)

        # Average train loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        #train_cos_sims.append(sum(epoch_cos_sims) / len(epoch_cos_sims))

        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Helps with memory fragmentation between processes

        # Validation phase
        ema_model.eval() # Using EMA, so not needed
        val_loss = 0
        #val_epoch_cos_sims = []
        with torch.no_grad():
            for images_in, images_out, conditions in val_loader:
                images_in, images_out, conditions = images_in.to(device), images_out.to(device), conditions.to(device)
                B = images_in.shape[0]
                reconstruction, global_step = ema_model(images_in, conditions)

                val_loss += criterion(reconstruction, images_out).item()#F.mse_loss(reconstruction, images_out).item()
                #cos_sim = F.cosine_similarity(reconstruction.view(B, -1), images_out.view(B, -1), dim=1).detach().mean().item()

                #val_epoch_cos_sims.append(cos_sim)

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        # Accumulate cosine similarities
        #val_cos_sims.append(sum(val_epoch_cos_sims) / len(val_epoch_cos_sims))

        lr_scheduler.step(val_loss)

        
        #val_sample = next(iter(val_loader))  # or fix one sample at the start and reuse
        save_debug_images(epoch, ema_model, val_sample, device)

        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Helps with memory fragmentation between processes

        end_time = time.time()
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")# | "
      #f"Train CosSim: {train_cos_sims[-1]:.4f}, Val CosSim: {val_cos_sims[-1]:.4f}")
        print(f"Epoch duration: {(end_time - start_time)/60:.2f} min")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        print(f"Current LPIPS weight: {lpips_weight:.2e}")


        # Save checkpoint every epoch (including loss history)
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "ema_state": ema_model.state_dict(),
            #"train_cos_sims": train_cos_sims,
            #"val_cos_sims": val_cos_sims,
            "lr_scheduler_state": lr_scheduler.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

        # Save best model separately
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            torch.save(ema_model.state_dict(), best_ema_model_path)
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    # Save final model after training completes
    torch.save(model.state_dict(), final_model_path)
    torch.save(ema_model.state_dict(), final_ema_model_path)

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("output/training_loss_plot.png")
    plt.show()

    #plt.figure(figsize=(10, 5))
    #plt.plot(train_cos_sims, label="Train Cosine Sim")
    #plt.plot(val_cos_sims, label="Val Cosine Sim")
    #plt.xlabel("Epoch")
    #plt.ylabel("Cosine Similarity")
    #plt.title("Training and Validation Cosine Similarity Over Epochs")
    #plt.legend()
    #plt.grid()
    #plt.savefig("output/cosine_similarity_plot.png")
    #plt.show()

    make_gif(vis_dir)


# Train the model
train(model, train_loader, val_loader, optimizer, device, num_epochs, val_sample)
