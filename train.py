import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from torchvision.models import ResNet18_Weights

# Dataset class
class StampDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, os.path.basename(self.dataframe.iloc[idx]['image']))
        image = Image.open(img_name).convert('RGB')
        label = int(self.dataframe.iloc[idx]['class']) - 1  # 0-indexed labels

        if self.transform:
            image = self.transform(image)

        return image, label


def train_model():
    # Load train, validation, and test CSV files
    train_df = pd.read_csv('dataset/train.csv')
    val_df = pd.read_csv('dataset/val.csv')

    # Data augmentations and transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets and DataLoaders
    train_dataset = StampDataset(train_df, 'dataset/train', transform=train_transform)
    val_dataset = StampDataset(val_df, 'dataset/validation', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(train_df['class'].unique()))  # Adjust for number of classes

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    model = model.to(device)

    # Loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, verbose=True)

    # Training parameters
    num_epochs = 20
    best_val_accuracy = 0.0
    patience = 3 # Number of epochs with no improvement to wait before stopping
    early_stopping_counter = 0  # Counter to track early stopping
    os.makedirs('models', exist_ok=True)  # Ensure the models directory exists

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Check for improvement
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), 'models/stamp_model.pth')
            print(f"Best model saved with Validation Accuracy: {best_val_accuracy:.4f}")
            early_stopping_counter = 0  # Reset the counter if improvement occurs
        else:
            early_stopping_counter += 1
            print(f"No improvement for {early_stopping_counter} epoch(s).")

        # Check if early stopping condition is met
        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break

        scheduler.step(val_acc)

    print("Training complete.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    train_model()
