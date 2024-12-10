import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from torchvision.models import ResNet18_Weights
import warnings

# Suppress FutureWarnings for trusted files
warnings.filterwarnings("ignore", category=FutureWarning)

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


def evaluate_model():
    # Load test dataset
    test_df = pd.read_csv('dataset/test.csv')
    
    # Transformations for test dataset
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Test dataset and DataLoader
    test_dataset = StampDataset(test_df, 'dataset/test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the model and adjust for the number of classes
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(test_df['class'].unique()))  # Adjust for number of classes

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load the best saved model weights
    model_path = 'models/stamp_model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model weights not found. Train the model first.")
    
    # Safely load weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Evaluation
    model.eval()
    test_corrects = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)

    test_acc = test_corrects.double() / len(test_loader.dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Manual Overfitting Check
    print("If the test accuracy is significantly lower than training and validation accuracy, the model may be overfitting.")


if __name__ == "__main__":
    evaluate_model()
