# coding=gb2312
import os
import time
import torch
import wandb
import math
import argparse
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, StepLR
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
from torchvision import transforms
from dataset import PigCoughDataset
from model import SelfNet, DeformableCNNBlock
from utils import compute_metrics, save_checkpoint, load_checkpoint, CustomLoss
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time 
from sklearn.metrics import precision_score, recall_score, average_precision_score
from sklearn.metrics import accuracy_score, f1_score




# Command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Training script for pig cough detection")
    parser.add_argument('-b', '--batch-size', type=int, default=16, help="Batch size for training")
    parser.add_argument('-a', '--smoothing-factor', type=float, default=0.2, help="Smoothing factor for custom loss")
    parser.add_argument('-l', '--learning-rate', type=float, default=3e-3, help="Initial learning rate")
    parser.add_argument('-i', '--save_index', type=int, default=0, help="Initial save_index")
    parser.add_argument('-s1o', '--stage_1_optimizer', type=str, default='SGD', help="stage_1_optimizer")
    parser.add_argument('-s2o', '--stage_2_optimizer', type=str, default='SGD', help="stage_2_optimizer")
    parser.add_argument('-w', '--weight_decay', type=float, default=0.01, help="weight_decay")
    parser.add_argument('-ir', '--ir_number', type=str, default='13.26', help="IR dataset identifier")
    parser.add_argument('-m', '--model-type', type=int, default=6, choices=[1, 2, 3, 4, 5, 6],
                        help="Select convolution type: 1=Standard, 2=Dilated, 3=DepthwiseSeparable, 4=Depthwise, 5=Pointwise, 6=Deformable")
    return parser.parse_args()

# Parse command-line arguments
args = parse_args()

# Configuration
EPOCHS_STAGE1 = 500 
EPOCHS_STAGE2 = 100 
BATCH_SIZE = args.batch_size 
LEARNING_RATE_STAGE1 = args.learning_rate  
IR = args.ir_number
LEARNING_RATE_STAGE2 = 1e-3  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "./data"
IMAGE_DIR = f"./images/Trainreshape-allpics6k-{IR}"
INDEX = args.save_index
RES_DIR = f"./res_{INDEX}"

STAGE_1_OPTIMIZER = args.stage_1_optimizer
STAGE_2_OPTIMIZER = args.stage_2_optimizer
WEIGHT_DECAY = args.weight_decay
SMOOTHING_FACTOR_ALPHA = args.smoothing_factor 

# WandB initialization
wandb.init(project="pig-cough-detection", config={
    "epochs_stage1": EPOCHS_STAGE1,
    "epochs_stage2": EPOCHS_STAGE2,
    "batch_size": BATCH_SIZE,
    "learning_rate_stage1": LEARNING_RATE_STAGE1,
    "learning_rate_stage2": LEARNING_RATE_STAGE2,
    "model_type": args.model_type
})

# Function to compute mean and std
def compute_mean_std(imgs_path):
    img_list = []
    imgs_path_list = os.listdir(imgs_path)
    for item in imgs_path_list:
        img = cv2.imread(os.path.join(imgs_path, item))
        img = cv2.resize(img, (100, 100))
        img_list.append(img)
    imgs = np.stack(img_list, axis=0).astype(np.float32) / 255.
    
    means, stdevs = [], []
    for i in range(3):  # Assuming BGR format, we compute mean and std for each channel
        pixels = imgs[:, :, :, i].ravel()  # Flatten all pixels in the channel
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    
    return means, stdevs

# Compute mean and std for current IR dataset
means, stdevs = compute_mean_std(IMAGE_DIR)

# Print the computed mean and std
print(f"Computed Mean: {means}")
print(f"Computed Std: {stdevs}")

# Transform with computed normalization values
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stdevs), 
])

# Dataset
dataset = PigCoughDataset(csv_file=f"{DATA_PATH}/outputB32-2546-{IR}IR.csv", image_dir=IMAGE_DIR, transform=transform)
kfold = KFold(n_splits=5, shuffle=True, random_state=None)

# Storage for F1 scores of all folds
fold_f1_scores = []

total_duration_all_folds = 0.0

# Define a function to save or update the best model for each stage and fold with overwrite
def save_or_update_best_model(model, optimizer, scheduler, epoch, train_loss, val_loss, f1_score, acc, precision, recall, auc_pr, learning_rate, fold, stage, highest_f1_score):
    if f1_score > highest_f1_score:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "f1_score": f1_score,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "auc_pr": auc_pr,
            "learning_rate": learning_rate,
            "fold": fold + 1
        }
        os.makedirs(RES_DIR, exist_ok=True)
        save_path = f"{RES_DIR}/best_model_fold_{fold + 1}_stage{stage}.pth"
        save_checkpoint(checkpoint, filename=save_path)
        
        # Save the best epoch's metrics in a text file
        best_epoch_info = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": acc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "auc_pr": auc_pr,
            "learning_rate": learning_rate
        }
        
        # Save the best epoch's information to a text file
        with open(f"{RES_DIR}/best_epoch_fold_{fold + 1}_stage{stage}.txt", "w") as f:
            for key, value in best_epoch_info.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Stage{stage} New best f1 score：{f1_score:.4f} at {epoch + 1} epoch，Model has been saved to {save_path}")
        return f1_score
    return highest_f1_score


def plot_metrics(fold, stage, train_losses, val_losses, accuracies, f1_scores, precisions, recalls, auc_prs):
    # Create a directory for saving plots if it doesn't exist
    os.makedirs(f'{RES_DIR}/plots', exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(20, 10))

    # Plot training and validation loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, 'r', label='Training Loss')
    plt.plot(epochs, val_losses, 'b', label='Validation Loss')
    plt.title(f'Fold {fold+1} Stage{stage} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(2, 3, 2)
    plt.plot(epochs, accuracies, 'g', label='Accuracy')
    plt.title(f'Fold {fold+1} Stage{stage} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot F1 Score
    plt.subplot(2, 3, 3)
    plt.plot(epochs, f1_scores, 'm', label='F1 Score')
    plt.title(f'Fold {fold+1} Stage{stage} F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    # --- Plot Precision ---
    plt.subplot(2, 3, 4)
    plt.plot(epochs, precisions, 'c', label='Precision')
    plt.title(f'Fold {fold+1} Stage{stage} Precision')
    plt.xlabel('Epochs'); plt.ylabel('Precision'); plt.legend()

    # --- Plot Recall ---
    plt.subplot(2, 3, 5)
    plt.plot(epochs, recalls, 'y', label='Recall')
    plt.title(f'Fold {fold+1} Stage{stage} Recall')
    plt.xlabel('Epochs'); plt.ylabel('Recall'); plt.legend()

    # --- Plot AUC-PR ---
    plt.subplot(2, 3, 6)
    plt.plot(epochs, auc_prs, 'k', label='AUC-PR')
    plt.title(f'Fold {fold+1} Stage{stage} AUC-PR')
    plt.xlabel('Epochs'); plt.ylabel('AUC-PR'); plt.legend()

    # Combine all plots into one figure
    plt.tight_layout()
    plt.savefig(f"{RES_DIR}/plots/fold_{fold+1}_stage{stage}_metrics.png")
    plt.close()
    




    
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    fold_start_time = time.time()
    print(f"\n===== Fold {fold + 1}/{kfold.n_splits} =====\n")
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

    
    model = SelfNet(model_type=args.model_type).to(DEVICE)
    
    # Stage 1: Joint training of the whole model using SGD optimizer with weight decay
    if STAGE_1_OPTIMIZER == 'SGD':
        optimizer_stage1 = optim.SGD(model.parameters(), lr=LEARNING_RATE_STAGE1, weight_decay=WEIGHT_DECAY)
    elif STAGE_1_OPTIMIZER == 'AdamW':
        optimizer_stage1 = optim.AdamW(model.parameters(), lr=LEARNING_RATE_STAGE1, weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported s1o")
    criterion_stage1 = nn.CrossEntropyLoss()
    
    # Initialize metrics lists for stage 1
    highest_f1_score_stage1 = 0.0
    train_losses_stage1 = []
    val_losses_stage1 = []
    accuracies_stage1 = []
    f1_scores_stage1 = []
    precisions_stage1 = []
    recalls_stage1 = []
    auc_prs_stage1 = []
    
    # Define Warm-up and Cosine Annealing Epochs
    warmup_epochs_stage1 = 20
    total_epochs_stage1 = EPOCHS_STAGE1

    # Define Warm-up scheduler
    warmup_scheduler_stage1 = LinearLR(optimizer_stage1, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs_stage1)

    if BATCH_SIZE == 16:
        T_max_tem = 69
    elif BATCH_SIZE == 32:
        T_max_tem = 35
    elif BATCH_SIZE == 64:
        T_max_tem = 18
    else:
      raise ValueError("Unsupported BATCH_SIZE")
    
    # Define Cosine Annealing scheduler
    cosine_scheduler_stage1 = CosineAnnealingLR(optimizer_stage1, T_max=T_max_tem * (total_epochs_stage1 - warmup_epochs_stage1), eta_min=1e-7)

    # Combine schedulers
    scheduler_stage1 = SequentialLR(optimizer_stage1, schedulers=[warmup_scheduler_stage1, cosine_scheduler_stage1], milestones=[warmup_epochs_stage1])

    print("Start Phase 1: Co-train the whole model")
    for epoch in range(EPOCHS_STAGE1):
        model.train()
        epoch_start = time.time()
        train_loss = 0

        # Training loop
        for images, labels in tqdm(train_loader, desc=f"Fold {fold +1} Stage1 Training Epoch {epoch + 1}/{EPOCHS_STAGE1}", colour="green"):
            images, labels = images.to(DEVICE), labels.to(DEVICE).long()
            optimizer_stage1.zero_grad()
            outputs = model(images)
            loss = criterion_stage1(outputs, labels)
            loss.backward()
            optimizer_stage1.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0
        y_true, y_pred, y_scores = [], [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating Stage1", colour="blue"):
                images, labels = images.to(DEVICE), labels.to(DEVICE).long()
                outputs = model(images)
                loss = criterion_stage1(outputs, labels)
                val_loss += loss.item()

          
                scores = torch.softmax(outputs, dim=1)[:, 1]

                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_scores.extend(scores.cpu().numpy())

        val_loss /= len(val_loader)
        try:
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            if len(set(y_true)) < 2:
                auc_pr = 0.0
                print("Warning: Only one class present in validation set. AUC-PR set to 0.0.")
            else:
                auc_pr = average_precision_score(y_true, y_scores)
        except Exception as e:
            print(f"Error computing metrics: {e}")
            acc, f1, precision, recall, auc_pr = 0.0, 0.0, 0.0, 0.0, 0.0

        # Step learning rate scheduler
        scheduler_stage1.step()

        # Logging
        wandb.log({
            f"fold_{fold}_stage1_train_loss": train_loss,
            f"fold_{fold}_stage1_val_loss": val_loss,
            f"fold_{fold}_stage1_accuracy": acc,
            f"fold_{fold}_stage1_f1_score": f1,
            f"fold_{fold}_stage1_precision": precision,
            f"fold_{fold}_stage1_recall": recall,
            f"fold_{fold}_stage1_auc_pr": auc_pr,
            f"fold_{fold}_stage1_learning_rate": scheduler_stage1.get_last_lr()[0] 
        })

        print(f"Stage1 - Epoch {epoch + 1} | Time: {time.time() - epoch_start:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | AUC-PR: {auc_pr:.4f}")
        
        # Update metrics lists
        train_losses_stage1.append(train_loss)
        val_losses_stage1.append(val_loss)
        accuracies_stage1.append(acc)
        f1_scores_stage1.append(f1)
        precisions_stage1.append(precision)
        recalls_stage1.append(recall)
        auc_prs_stage1.append(auc_pr)

        # Check and update the best model based on F1 score for stage 1
        highest_f1_score_stage1 = save_or_update_best_model(
            model=model,
            optimizer=optimizer_stage1,
            scheduler=scheduler_stage1,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            f1_score=f1,
            acc=acc,
            precision=precision,
            recall=recall,
            auc_pr=auc_pr,
            learning_rate=scheduler_stage1.get_last_lr()[0],
            fold=fold,
            stage=1,
            highest_f1_score=highest_f1_score_stage1
        )

    plot_metrics(fold=fold, stage=1, train_losses=train_losses_stage1, val_losses=val_losses_stage1,
                 accuracies=accuracies_stage1, f1_scores=f1_scores_stage1, 
                 precisions=precisions_stage1, recalls=recalls_stage1, auc_prs=auc_prs_stage1)

    # Stage 2: Fine-tuning only the classifier part
    print("Start Phase 2: Freeze the feature extractor and train only the classifier")
    model.freeze_feature_extractor()
    
    # Define optimizer for classifier parameters only
    if STAGE_2_OPTIMIZER == 'SGD':
        optimizer_stage2 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE_STAGE2, weight_decay=WEIGHT_DECAY)
    elif STAGE_2_OPTIMIZER == 'AdamW':
        optimizer_stage2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE_STAGE2, weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported s1o")

    criterion_stage2 = nn.CrossEntropyLoss().to(DEVICE)
    
    # Define learning rate scheduler, using StepLR
    scheduler_stage2 = StepLR(optimizer_stage2, step_size=30, gamma=0.1)
    
    # Initialize metrics lists for stage 2
    highest_f1_score_stage2 = 0.0
    train_losses_stage2 = []
    val_losses_stage2 = []
    accuracies_stage2 = []
    f1_scores_stage2 = []
    precisions_stage2 = []
    recalls_stage2 = []
    auc_prs_stage2 = []

    for epoch in range(EPOCHS_STAGE2):
        model.train()
        epoch_start = time.time()
        train_loss = 0

        # Training loop
        for images, labels in tqdm(train_loader, desc=f"Fold {fold +1} Stage2 Training Epoch {epoch + 1}/{EPOCHS_STAGE2}", colour="green"):
            images, labels = images.to(DEVICE), labels.to(DEVICE).long()
            optimizer_stage2.zero_grad()
            outputs = model(images)
            loss = criterion_stage2(outputs, labels)
            loss.backward()
            optimizer_stage2.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0
        y_true, y_pred, y_scores = [], [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating Stage2", colour="blue"):
                images, labels = images.to(DEVICE), labels.to(DEVICE).long()
                outputs = model(images)
                loss = criterion_stage2(outputs, labels)
                val_loss += loss.item()

       
                scores = torch.softmax(outputs, dim=1)[:, 1]

                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_scores.extend(scores.cpu().numpy())

        val_loss /= len(val_loader)
        try:
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            if len(set(y_true)) < 2:
                auc_pr = 0.0
                print("Warning: Only one class present in validation set. AUC-PR set to 0.0.")
            else:
                auc_pr = average_precision_score(y_true, y_scores)
        except Exception as e:
            print(f"Error computing metrics: {e}")
            acc, f1, precision, recall, auc_pr = 0.0, 0.0, 0.0, 0.0, 0.0

        # Step learning rate scheduler
        scheduler_stage2.step()

        # Logging
        wandb.log({
            f"fold_{fold}_stage2_train_loss": train_loss, 
            f"fold_{fold}_stage2_val_loss": val_loss,
            f"fold_{fold}_stage2_accuracy": acc,
            f"fold_{fold}_stage2_f1_score": f1,
            f"fold_{fold}_stage2_precision": precision,
            f"fold_{fold}_stage2_recall": recall,
            f"fold_{fold}_stage2_auc_pr": auc_pr,
            f"fold_{fold}_stage2_learning_rate": scheduler_stage2.get_last_lr()[0] 
        })

        print(f"Stage2 - Epoch {epoch + 1} | Time: {time.time() - epoch_start:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | AUC-PR: {auc_pr:.4f}")
        
        # Update metrics lists
        train_losses_stage2.append(train_loss)
        val_losses_stage2.append(val_loss)
        accuracies_stage2.append(acc)
        f1_scores_stage2.append(f1)
        precisions_stage2.append(precision)
        recalls_stage2.append(recall)
        auc_prs_stage2.append(auc_pr)

        # Check and update the best model based on F1 score for stage 2
        highest_f1_score_stage2 = save_or_update_best_model(
            model=model,
            optimizer=optimizer_stage2,
            scheduler=scheduler_stage2,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            f1_score=f1,
            acc=acc,
            precision=precision,
            recall=recall,
            auc_pr=auc_pr,
            learning_rate=scheduler_stage2.get_last_lr()[0],
            fold=fold,
            stage=2,
            highest_f1_score=highest_f1_score_stage2
        )

    plot_metrics(fold=fold, stage=2, train_losses=train_losses_stage2, val_losses=val_losses_stage2,
                 accuracies=accuracies_stage2, f1_scores=f1_scores_stage2,
                 precisions=precisions_stage2, recalls=recalls_stage2, auc_prs=auc_prs_stage2)

    fold_end_time = time.time()
    fold_duration = fold_end_time - fold_start_time
    total_duration_all_folds += fold_duration
    print(f"\n===== Fold {fold + 1} Training completed, total time: {fold_duration:.2f}s =====\n")
    wandb.log({f"fold_{fold}_total_duration": fold_duration})
    
    # Collect F1 scores from stage 2 for average calculation
    fold_f1_scores.append(highest_f1_score_stage2)

# Compute and log the average F1 Score across all folds
average_f1_score = sum(fold_f1_scores) / len(fold_f1_scores)
print(f"\n===== 5-Fold Cross Validation Completed =====")
print(f"Average F1 Score: {average_f1_score:.4f}")
wandb.log({"average_f1_score": average_f1_score})

print(f"\nTotal training time for 5 folds: {total_duration_all_folds:.2f}s")

os.makedirs(RES_DIR, exist_ok=True)

total_time_file = os.path.join(RES_DIR, "total_time.txt")

try:
    with open(total_time_file, 'w', encoding='utf-8') as f:
        hours = int(total_duration_all_folds // 3600)
        minutes = int((total_duration_all_folds % 3600) // 60)
        seconds = total_duration_all_folds % 60
        f.write(f"Total training time for 5 folds: {total_duration_all_folds:.2f} seconds\n")
        f.write(f"Which is: {hours:02d}h {minutes:02d}m {seconds:05.2f}s\n")
    print(f"Total time saved to {total_time_file}")
except Exception as e:
    print(f"Error saving total time to file: {e}")
wandb.finish()