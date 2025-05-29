import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time

# --- Configuration ---
DATA_DIR = 'dataset' # Replace with the actual path to your dataset folder
MODEL_SAVE_PATH = 'best_model.pth'
NUM_CLASSES = 4 # freshapples, freshoranges, rottenapples, rottenoranges
BATCH_SIZE = 32
NUM_EPOCHS = 100 # Set a reasonably high number, early stopping will stop it
LEARNING_RATE = 0.001
PATIENCE = 10 # Number of epochs to wait for validation loss improvement before stopping
CHECKPOINT_INTERVAL = 5 # Save checkpoint every N epochs (optional)
WEIGHT_DECAY = 1e-5 # L2 regularization to help prevent overfitting
NUM_WORKERS = 4 # Number of subprocesses for data loading. Set to 0 if you still have issues.

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Transformations ---
# We'll use standard ImageNet normalization as we're using a pre-trained ImageNet model
# Mean and standard deviation values for ImageNet
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Training transforms with augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224), # Random crop and resize
    transforms.RandomHorizontalFlip(), # Random horizontal flip
    transforms.RandomRotation(10), # Random rotation up to 10 degrees
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Minor color jitter
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std) # Normalize
])

# Validation and Test transforms (no aggressive augmentation)
val_test_transforms = transforms.Compose([
    transforms.Resize(256), # Resize to 256
    transforms.CenterCrop(224), # Crop to 224x224
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std) # Normalize
])

# --- Load Data ---
def load_data(data_dir, train_transforms, val_test_transforms, batch_size, num_workers, num_classes):
    print("Loading datasets...")
    try:
        image_datasets = {
            'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms),
            'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), val_test_transforms),
            'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), val_test_transforms)
        }

        # Check class names and mapping
        class_names = image_datasets['train'].classes
        print(f"Found class names: {class_names}")
        if len(class_names) != num_classes:
             print(f"Warning: Found {len(class_names)} classes, expected {num_classes}. Adjusting NUM_CLASSES.")
             num_classes = len(class_names) # Adjust NUM_CLASSES if necessary


        # --- Check for and Handle Data Imbalance ---
        print("Checking for data imbalance...")
        train_targets = image_datasets['train'].targets
        class_counts = np.bincount(train_targets)
        print(f"Class counts in training set: {dict(zip(class_names, class_counts))}")

        total_samples = len(train_targets)
        if np.min(class_counts) == 0:
             print("FATAL ERROR: One or more classes have zero training samples!")
             exit() # Cannot train if a class has no data

        # Calculate class weights inversely proportional to class frequencies
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
        # weights = class_weights[train_targets] # Not used with weighted loss, but kept for reference if using WeightedRandomSampler

        # DataLoaders
        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
            'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
            'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        }
        print("Data loaders created.")
        return dataloaders, class_names, class_weights, num_classes

    except FileNotFoundError:
        print(f"FATAL ERROR: Data directory not found at {data_dir}")
        print("Please check the DATA_DIR path configuration.")
        exit()
    except Exception as e:
        print(f"FATAL ERROR during data loading: {e}")
        exit()

# --- Model Setup ---
def setup_model(num_classes, device):
    print("Setting up model...")
    # Load a pre-trained ResNet-50 model
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT') # Use weights=None for no pre-training

    # Freeze all layers in the pre-trained model initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)
    print("Model loaded and modified.")
    return model

# --- Training and Validation Function ---
def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs, patience, model_save_path, device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0 # Counter for early stopping

    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            total_samples_phase = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples_phase += inputs.size(0)

            epoch_loss = running_loss / total_samples_phase
            epoch_acc = running_corrects.double() / total_samples_phase

            history[f'{phase}_loss'].append(epoch_loss)
            if phase == 'val':
                 history[f'{phase}_accuracy'].append(epoch_acc.item())


            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Step the scheduler based on validation loss
            if phase == 'val':
                scheduler.step(epoch_loss)

                # Early stopping logic
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print(f"Validation loss improved. Saving best model weights to {model_save_path}")
                    torch.save(best_model_wts, model_save_path)
                else:
                    epochs_no_improve += 1
                    print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")
                    if epochs_no_improve >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        time_elapsed = time.time() - since
                        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                        # Load best model weights before returning
                        model.load_state_dict(best_model_wts)
                        return model, history # Stop training

        # Optional: Save checkpoint periodically
        # if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
        #     checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'best_loss': best_loss,
        #         'epochs_no_improve': epochs_no_improve,
        #         'history': history
        #     }, checkpoint_path)
        #     print(f"Checkpoint saved to {checkpoint_path}")


        print() # Newline after each epoch

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# --- Plot Training History ---
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs. Epoch')
    plt.legend()

    # plt.show() # Comment out plt.show() if running in an environment where it blocks execution

# --- Evaluate on Test Set ---
def evaluate_model(model, dataloader, class_names, device):
    model.eval() # Set model to evaluate mode

    running_corrects = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    print("Evaluating model on test set...")
    with torch.no_grad(): # No gradient calculation needed
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

    test_acc = running_corrects.double() / total_samples
    print(f'Test Accuracy: {test_acc:.4f}')

    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Test Set)')
    # plt.show() # Comment out plt.show() if running in an environment where it blocks execution

    # Generate Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report (Test Set):\n")
    print(report)


# --- Main Execution Block ---
# This ensures the code inside only runs when the script is executed directly
if __name__ == '__main__':
    # Added freeze_support() for better compatibility on Windows when freezing (optional but good practice)
    # from multiprocessing import freeze_support
    # freeze_support()

    dataloaders, class_names, class_weights, NUM_CLASSES = load_data(DATA_DIR, train_transforms, val_test_transforms, BATCH_SIZE, NUM_WORKERS, NUM_CLASSES)

    model = setup_model(NUM_CLASSES, device)

    # --- Loss Function, Optimizer, Scheduler ---
    # Use CrossEntropyLoss with class weights calculated earlier to handle imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Reduce learning rate when validation loss has stopped improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


    # --- Start Training ---
    print("Starting model training...")
    model_ft, history = train_model(model, criterion, optimizer, scheduler, dataloaders, NUM_EPOCHS, PATIENCE, MODEL_SAVE_PATH, device)

    # --- Plot Training History ---
    print("Plotting training history...")
    plot_history(history)
    plt.show() # Show plots here after training is done

    # --- Evaluate on Test Set ---
    evaluate_model(model_ft, dataloaders['test'], class_names, device)
    plt.show() # Show confusion matrix plot here

    print("\nTraining and evaluation complete.")

