# coding=gb2312
import os
import time
import torch
import wandb
import math
import argparse
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, StepLR
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import KFold 
from tqdm import tqdm
from torchvision import transforms
from dataset import PigCoughDataset
from model import SelfNet, DeformableCNNBlock
from utils import compute_metrics, save_checkpoint, load_checkpoint, CustomLoss
import matplotlib.pyplot as plt
import numpy as np
import cv2
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
    parser.add_argument('-w', '--weight_decay', type=float, default=0.01, help="weight_decay")
    parser.add_argument('-ir', '--ir_number', type=str, default='13.26', help="IR dataset identifier")
    parser.add_argument('-m', '--model-type', type=int, default=6, choices=[1, 2, 3, 4, 5, 6],
                        help="Select convolution type: 1=Standard, 2=Dilated, 3=DepthwiseSeparable, 4=Depthwise, 5=Pointwise, 6=Deformable")
    return parser.parse_args()

# Parse command-line arguments
args = parse_args()

# Configuration
EPOCHS = 500 
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
IR = args.ir_number
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "./data"
IMAGE_DIR = f"./images/Trainreshape-allpics6k-{IR}"
INDEX = args.save_index
RES_DIR = f"./test_{INDEX}"

OPTIMIZER_TYPE = args.stage_1_optimizer 
WEIGHT_DECAY = args.weight_decay
SMOOTHING_FACTOR_ALPHA = args.smoothing_factor

# WandB initialization
wandb.init(project="pig-cough-detection-single-run", config={
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "model_type": args.model_type,
    "optimizer": OPTIMIZER_TYPE,
    "weight_decay": WEIGHT_DECAY,
    "smoothing_factor": SMOOTHING_FACTOR_ALPHA
})

# Function to compute mean and std
def compute_mean_std(imgs_path):
    img_list = []
    imgs_path_list = os.listdir(imgs_path)
    for item in imgs_path_list:
        img = cv2.imread(os.path.join(imgs_path, item))
        if img is not None: 
            img = cv2.resize(img, (100, 100))
            img_list.append(img)
    
    if not img_list:
        raise ValueError(f"No images found or loaded from {imgs_path} to compute mean and std.")

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
full_dataset = PigCoughDataset(csv_file=f"{DATA_PATH}/outputB32-2546-{IR}IR.csv", image_dir=IMAGE_DIR, transform=transform)

train_val_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_val_size
train_val_dataset, test_dataset = random_split(full_dataset, [train_val_size, test_size], generator=torch.Generator().manual_seed(42)) 


train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))


print(f"Data set size: total {len(full_dataset)} | Train_set {len(train_dataset)} | Val_set {len(val_dataset)} | Test_set {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define a function to save or update the best model with overwrite
def save_or_update_best_model(model, optimizer, scheduler, epoch, train_loss, val_loss, f1_score, acc, precision, recall, auc_pr, learning_rate, highest_f1_score):
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
        }
        os.makedirs(RES_DIR, exist_ok=True)
        save_path = f"{RES_DIR}/best_model.pth" # 不再有fold和stage
        save_checkpoint(checkpoint, filename=save_path)
        
        # Save the best epoch's metrics in a text file
        best_epoch_info = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": acc,
            "f1_score": f1_score,
            "precision": precision,
            "recall": recall,
            "auc_pr": auc_pr,
            "learning_rate": learning_rate
        }
        
        # Save the best epoch's information to a text file
        with open(f"{RES_DIR}/best_epoch_info.txt", "w") as f:
            for key, value in best_epoch_info.items():
                f.write(f"{key}: {value}\n")
        
        print(f"New best F1 score：{f1_score:.4f} at {epoch + 1} epoch，Model has been saved to {save_path}")
        return f1_score
    return highest_f1_score


def plot_metrics(train_losses, val_losses, accuracies, f1_scores, precisions, recalls, auc_prs):
    # Create a directory for saving plots if it doesn't exist
    os.makedirs(f'{RES_DIR}/plots', exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(20, 10))

    # Plot training and validation loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, 'r', label='Training Loss')
    plt.plot(epochs, val_losses, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(2, 3, 2)
    plt.plot(epochs, accuracies, 'g', label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot F1 Score
    plt.subplot(2, 3, 3)
    plt.plot(epochs, f1_scores, 'm', label='F1 Score')
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    # --- Plot Precision ---
    plt.subplot(2, 3, 4)
    plt.plot(epochs, precisions, 'c', label='Precision')
    plt.title('Precision')
    plt.xlabel('Epochs'); plt.ylabel('Precision'); plt.legend()

    # --- Plot Recall ---
    plt.subplot(2, 3, 5)
    plt.plot(epochs, recalls, 'y', label='Recall')
    plt.title('Recall')
    plt.xlabel('Epochs'); plt.ylabel('Recall'); plt.legend()

    # --- Plot AUC-PR ---
    plt.subplot(2, 3, 6)
    plt.plot(epochs, auc_prs, 'k', label='AUC-PR')
    plt.title('AUC-PR')
    plt.xlabel('Epochs'); plt.ylabel('AUC-PR'); plt.legend()

    # Combine all plots into one figure
    plt.tight_layout()
    plt.savefig(f"{RES_DIR}/plots/metrics.png") 
    plt.close()
    


model = SelfNet(model_type=args.model_type).to(DEVICE)


if OPTIMIZER_TYPE == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER_TYPE == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
else:
    raise ValueError("Unsupported optimizer type. Choose 'SGD' or 'AdamW'.")

criterion = nn.CrossEntropyLoss().to(DEVICE)

# Initialize metrics lists
highest_f1_score = 0.0
train_losses = []
val_losses = []
accuracies = []
f1_scores = []
precisions = []
recalls = []
auc_prs = []

# Define Warm-up and Cosine Annealing Epochs
warmup_epochs = 20
total_epochs = EPOCHS

# Define Warm-up scheduler
warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)

if BATCH_SIZE == 16:
    T_max_tem = 69
elif BATCH_SIZE == 32:
    T_max_tem = 35
elif BATCH_SIZE == 64:
    T_max_tem = 18
else:

    T_max_tem = int(len(train_loader) * (total_epochs - warmup_epochs))
    print(f"Warning: Using dynamically calculated T_max_tem = {T_max_tem} for CosineAnnealingLR.")

# Define Cosine Annealing scheduler
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max_tem * (total_epochs - warmup_epochs), eta_min=1e-7)

# Combine schedulers
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

print("Begin training")
training_start_time = time.time() 

for epoch in range(EPOCHS):
    model.train()
    epoch_start = time.time()
    train_loss = 0

    # Training loop
    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}", colour="green"):
        images, labels = images.to(DEVICE), labels.to(DEVICE).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", colour="blue"):
            images, labels = images.to(DEVICE), labels.to(DEVICE).long()
            outputs = model(images)
            loss = criterion(outputs, labels)
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
    scheduler.step()

    # Logging
    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "auc_pr": auc_pr,
        "learning_rate": scheduler.get_last_lr()[0]
    })

    print(f"Epoch {epoch + 1} | Time: {time.time() - epoch_start:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | AUC-PR: {auc_pr:.4f}")
    
    # Update metrics lists
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    accuracies.append(acc)
    f1_scores.append(f1)
    precisions.append(precision)
    recalls.append(recall)
    auc_prs.append(auc_pr)

    # Check and update the best model based on F1 score
    highest_f1_score = save_or_update_best_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        train_loss=train_loss,
        val_loss=val_loss,
        f1_score=f1,
        acc=acc,
        precision=precision,
        recall=recall,
        auc_pr=auc_pr,
        learning_rate=scheduler.get_last_lr()[0],
        highest_f1_score=highest_f1_score
    )

training_end_time = time.time()
training_duration = training_end_time - training_start_time
print(f"\n Training completed, total time: {training_duration:.2f}s")
wandb.log({"total_training_duration": training_duration})


plot_metrics(train_losses=train_losses, val_losses=val_losses,
             accuracies=accuracies, f1_scores=f1_scores,
             precisions=precisions, recalls=recalls, auc_prs=auc_prs)


print("\n===== Begin evaluating the best model on an independent test set =====")


best_model_path = f"{RES_DIR}/best_model.pth"
if not os.path.exists(best_model_path):
    print("Error: The best model file was not found for testing. Please check if the training process saved the model successfully")
    wandb.finish()
    exit()

model_for_test = SelfNet(model_type=args.model_type).to(DEVICE)
checkpoint = torch.load(best_model_path, map_location=DEVICE)
model_for_test.load_state_dict(checkpoint['state_dict'])
model_for_test.eval()

test_start_time = time.time()
test_loss = 0
y_true_test, y_pred_test, y_scores_test = [], [], []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing on independent set", colour="red"):
        images, labels = images.to(DEVICE), labels.to(DEVICE).long()
        outputs = model_for_test(images)
        loss = criterion(outputs, labels) 
        test_loss += loss.item()

        scores = torch.softmax(outputs, dim=1)[:, 1]
        _, preds = torch.max(outputs, 1)

        y_true_test.extend(labels.cpu().numpy())
        y_pred_test.extend(preds.cpu().numpy())
        y_scores_test.extend(scores.cpu().numpy())

test_end_time = time.time()
total_test_duration = test_end_time - test_start_time

test_loss /= len(test_loader)
try:
    test_acc = accuracy_score(y_true_test, y_pred_test)
    test_f1 = f1_score(y_true_test, y_pred_test, average='macro', zero_division=0)
    test_precision = precision_score(y_true_test, y_pred_test, average='macro', zero_division=0)
    test_recall = recall_score(y_true_test, y_pred_test, average='macro', zero_division=0)
    if len(set(y_true_test)) < 2:
        test_auc_pr = 0.0
        print("Warning: there is only one category in the test set. the AUC-PR is set to 0.0。")
    else:
        test_auc_pr = average_precision_score(y_true_test, y_scores_test)
except Exception as e:
    print(f"Error calculating test metrics. {e}")
    test_acc, test_f1, test_precision, test_recall, test_auc_pr = 0.0, 0.0, 0.0, 0.0, 0.0

print(f"\nIndependent test set results:")
print(f"Test set sample size: {len(test_dataset)}")
print(f"Total test set run time: {total_test_duration:.2f}s")
print(f"test loss: {test_loss:.4f} | acc: {test_acc:.4f} | F1: {test_f1:.4f} | precision: {test_precision:.4f} | recall: {test_recall:.4f} | AUC-PR: {test_auc_pr:.4f}")


os.makedirs(RES_DIR, exist_ok=True)
test_results_file = os.path.join(RES_DIR, "test_results.txt")

try:
    with open(test_results_file, 'w', encoding='utf-8') as f:
        f.write(f"final_test_samples: {len(test_dataset)}\n")
        f.write(f"final_test_duration_seconds: {total_test_duration:.2f}s\n")
        f.write(f"final_test_loss: {test_loss:.4f}\n")
        f.write(f"final_test_accuracy: {test_acc:.4f}\n")
        f.write(f"final_test_f1_score: {test_f1:.4f}\n")
        f.write(f"final_test_precision: {test_precision:.4f}\n")
        f.write(f"final_test_recall: {test_recall:.4f}\n")
        f.write(f"final_test_auc_pr: {test_auc_pr:.4f}\n")
    print(f"The test results have been saved to {test_results_file}")
except Exception as e:
    print(f"Error saving test results to file: {e}")


wandb.log({
    "final_test_samples": len(test_dataset),
    "final_test_duration_seconds": total_test_duration,
    "final_test_loss": test_loss,
    "final_test_accuracy": test_acc,
    "final_test_f1_score": test_f1,
    "final_test_precision": test_precision,
    "final_test_recall": test_recall,
    "final_test_auc_pr": test_auc_pr
})

wandb.finish()