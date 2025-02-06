import os
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
 
# Enable cuDNN benchmarking for better performance on GPUs
torch.backends.cudnn.benchmark = True
 
# Define custom dataset class to load defect images and labels
class DefectDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # Folder names represent defect classes
        self.file_paths = []
        self.labels = []
 
        for label, defect_class in enumerate(self.classes):
            defect_dir = os.path.join(root_dir, defect_class)

            for img_file in os.listdir(defect_dir):
                self.file_paths.append(os.path.join(defect_dir, img_file))
                self.labels.append(label)  # Assign label based on folder index
 
    def __len__(self):
        return len(self.file_paths)
 
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
 
# Set up the data directory
data_dir = "data/training_sessions/BEOL_CMP_new_trng/images"
 
# Define transformations to resize all images to a consistent size
transform = transforms.Compose([
    transforms.Resize((960, 1440)),  # Resize to height=960, width=1440
    transforms.ToTensor()           # Convert images to tensors
])
 
# Create dataset
dataset = DefectDataset(data_dir, transform=transform)
 
# Split dataset into training, validation, and test sets
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)  # 80% for training
val_size = int(0.1 * dataset_size)    # 10% for validation
test_size = dataset_size - train_size - val_size  # Remaining 10% for testing
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
 
# Create DataLoaders for each split
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)
 
# Define the defect classification model with two branches
class DefectClassifier(nn.Module):
    def __init__(self, pretrained_model):
        super(DefectClassifier, self).__init__()
        self.branch1 = nn.Sequential(
            pretrained_model,
            nn.Linear(1000, 2)  # Output: 2 classes
        )

        self.branch2 = nn.Sequential(
            pretrained_model,
            nn.Linear(1000, 23)  # Output: 23 classes
        )
 
    def forward(self, x):
        def resize_segment(segment):
            return F.interpolate(segment, size=(224, 224), mode='bilinear', align_corners=False)
 
        _, _, height, width = x.shape
    
        if height == 960 and width == 1440:
            first_4_segments = resize_segment(x[:, :, :960, :960])
            last_2_segments = resize_segment(x[:, :, :960, 960:1440])
            out1 = self.branch1(first_4_segments)
            out2 = self.branch2(last_2_segments)
            return out1, out2

        elif height == 1440 and width == 1440:
            optical_segment = resize_segment(x[:, :, 960:1440, :480])
            out1 = torch.zeros(x.size(0), 2).to(x.device)
            out2 = torch.full((x.size(0), 23), -float('inf')).to(x.device)
            out2[:, 0] = 0
            return out1, out2

        else:
            raise ValueError(f"Unexpected image dimensions: {height}x{width}")
 
# Load pretrained ConViT model
pretrained_model = timm.create_model('convit_base', pretrained=False, num_classes=1000)
state_dict = torch.load('./convit_base.fb_in1k/pytorch_model.bin', map_location=torch.device('cpu'))
pretrained_model.load_state_dict(state_dict)
 
# Fix: Define `device` before moving the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Added this line
 
# Initialize the defect classifier model
model = DefectClassifier(pretrained_model)
model = model.to(device)  # Now this works because `device` is defined
 
# Define loss function and optimizer
criterion1 = nn.BCEWithLogitsLoss()
criterion2 = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
 
# Training loop
num_epochs = 85  # Updated the number of epochs

for epoch in range(num_epochs):
    # Training phase
    model.train()

    total_loss, correct_branch1, correct_branch2, total_samples = 0, 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        batch_size = labels.size(0)
        total_samples += batch_size
 
        # Forward pass
        out1, out2 = model(images)
        defect_labels = (labels != 0).long()
        defect_type_labels = labels
        defect_labels = F.one_hot(defect_labels, num_classes=2).float()
 
        # Compute losses
        loss1 = criterion1(out1, defect_labels)
        loss2 = criterion2(out2, defect_type_labels)
        loss = loss1 + loss2
 
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred1 = torch.max(out1, 1)
        correct_branch1 += (pred1 == defect_labels.argmax(dim=1)).sum().item()
        _, pred2 = torch.max(out2, 1)
        correct_branch2 += (pred2 == defect_type_labels).sum().item()
 
    avg_loss = total_loss / len(train_loader)
    accuracy_branch1 = correct_branch1 / total_samples * 100
    accuracy_branch2 = correct_branch2 / total_samples * 100

    print(f"Epoch [{epoch+1}/{num_epochs}] Training - Loss: {avg_loss:.4f}, Branch 1 Acc: {accuracy_branch1:.2f}%, Branch 2 Acc: {accuracy_branch2:.2f}%")
 
    # Validation phase
    model.eval()
    val_loss, val_correct_branch1, val_correct_branch2, val_samples = 0, 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            val_samples += labels.size(0)
            out1, out2 = model(images)
            
            defect_labels = (labels != 0).long()
            defect_type_labels = labels
            defect_labels = F.one_hot(defect_labels, num_classes=2).float()
            
            loss1 = criterion1(out1, defect_labels)
            loss2 = criterion2(out2, defect_type_labels)

            val_loss += loss1.item() + loss2.item()
            _, pred1 = torch.max(out1, 1)

            val_correct_branch1 += (pred1 == defect_labels.argmax(dim=1)).sum().item()
            _, pred2 = torch.max(out2, 1)

            val_correct_branch2 += (pred2 == defect_type_labels).sum().item()
 
    val_avg_loss = val_loss / len(val_loader)
    val_accuracy_branch1 = val_correct_branch1 / val_samples * 100
    val_accuracy_branch2 = val_correct_branch2 / val_samples * 100

    print(f"Epoch [{epoch+1}/{num_epochs}] Validation - Loss: {val_avg_loss:.4f}, Branch 1 Acc: {val_accuracy_branch1:.2f}%, Branch 2 Acc: {val_accuracy_branch2:.2f}%")
 
# Test phase (run after training)
model.eval()
test_loss, test_correct_branch1, test_correct_branch2, test_samples = 0, 0, 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        test_samples += labels.size(0)
        out1, out2 = model(images)
        
        defect_labels = (labels != 0).long()
        defect_type_labels = labels
        defect_labels = F.one_hot(defect_labels, num_classes=2).float()
 
        loss1 = criterion1(out1, defect_labels)
        loss2 = criterion2(out2, defect_type_labels)
        test_loss += loss1.item() + loss2.item()
 
        _, pred1 = torch.max(out1, 1)
        test_correct_branch1 += (pred1 == defect_labels.argmax(dim=1)).sum().item()

        _, pred2 = torch.max(out2, 1)
        test_correct_branch2 += (pred2 == defect_type_labels).sum().item()
 
test_avg_loss = test_loss / len(test_loader)
test_accuracy_branch1 = test_correct_branch1 / test_samples * 100
test_accuracy_branch2 = test_correct_branch2 / test_samples * 100

print(f"Test Results - Loss: {test_avg_loss:.4f}, Branch 1 Acc: {test_accuracy_branch1:.2f}%, Branch 2 Acc: {test_accuracy_branch2:.2f}%")
 
# Save the trained model
torch.save(model.state_dict(), "model_new.pth")
print("Model saved to 'model_new.pth'.")
