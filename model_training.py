import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time

DATA_DIR = 'dataset'
MODEL_SAVE_PATH = 'best_model.pth'
NUM_CLASSES = 4
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10
CHECKPOINT_INTERVAL = 5
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

def load_data(data_dir, train_transforms, val_test_transforms, batch_size, num_workers, num_classes):
    print("Loading datasets...")
    try:
        image_datasets = {
            'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms),
            'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), val_test_transforms),
            'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), val_test_transforms)
        }

        class_names = image_datasets['train'].classes
        print(f"Found class names: {class_names}")

        if len(class_names) != num_classes:
            print(f"Warning: Expected {num_classes} classes, but found {len(class_names)}. Adjusting.")
            num_classes = len(class_names)

        print("Checking for data imbalance...")
        train_targets = image_datasets['train'].targets
        class_counts = np.bincount(train_targets)
        print(f"Class counts in training set: {dict(zip(class_names, class_counts))}")

        if np.min(class_counts) == 0:
            print("FATAL ERROR: One or more classes have zero training samples!")
            exit()

        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)

        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
            'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
            'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        }
        print("Data loaders created.")
        return dataloaders, class_names, class_weights, num_classes

    except FileNotFoundError:
        print(f"FATAL ERROR: Data directory not found at {data_dir}")
        exit()
    except Exception as e:
        print(f"FATAL ERROR during data loading: {e}")
        exit()

def setup_model(num_classes, device):
    print("Setting up model...")
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    print("Model loaded and modified.")
    return model

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs, patience, model_save_path, device):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}\n' + '-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            total_samples_phase = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples_phase += inputs.size(0)

            epoch_loss = running_loss / total_samples_phase
            epoch_acc = running_corrects.double() / total_samples_phase

            history[f'{phase}_loss'].append(epoch_loss)
            if phase == 'val':
                history['val_accuracy'].append(epoch_acc.item())
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print(f"Validation loss improved. Saving model to {model_save_path}")
                    torch.save(best_model_wts, model_save_path)
                else:
                    epochs_no_improve += 1
                    print(f"No improvement for {epochs_no_improve} epoch(s).")
                    if epochs_no_improve >= patience:
                        print(f"Early stopping after {epoch + 1} epochs.")
                        model.load_state_dict(best_model_wts)
                        return model, history
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    model.load_state_dict(best_model_wts)
    return model, history

def plot_history(history, save_dir='figures'):
    os.makedirs(save_dir, exist_ok=True)

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

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def evaluate_model(model, dataloader, class_names, device, save_dir='figures'):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    all_preds = []
    all_labels = []
    running_corrects = 0
    total_samples = 0
    print("Evaluating model on test set...")
    with torch.no_grad():
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
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Test Set)')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report (Test Set):\n")
    print(report)

if __name__ == '__main__':
    dataloaders, class_names, class_weights, NUM_CLASSES = load_data(DATA_DIR, train_transforms, val_test_transforms, BATCH_SIZE, NUM_WORKERS, NUM_CLASSES)
    model = setup_model(NUM_CLASSES, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    print("Starting model training...")
    model_ft, history = train_model(model, criterion, optimizer, scheduler, dataloaders, NUM_EPOCHS, PATIENCE, MODEL_SAVE_PATH, device)
    print("Plotting training history...")
    plot_history(history)
    print("Evaluating model and saving confusion matrix...")
    evaluate_model(model_ft, dataloaders['test'], class_names, device)
    print("\nTraining and evaluation complete.")
