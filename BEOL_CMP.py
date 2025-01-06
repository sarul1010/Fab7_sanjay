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
        """
        Args:
            root_dir (str): Path to the root directory containing class folders.
            transform (callable, optional): Optional transforms to apply to images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # Folder names represent defect classes
        self.file_paths = []
        self.labels = []

        # Iterate through all class folders and load image paths with corresponding labels
        for label, defect_class in enumerate(self.classes):
            defect_dir = os.path.join(root_dir, defect_class)
            for img_file in os.listdir(defect_dir):
                self.file_paths.append(os.path.join(defect_dir, img_file))
                self.labels.append(label)  # Assign label based on folder index

    def __len__(self):
        return len(self.file_paths)  # Return total number of images
 
    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image.
            label: Numeric label corresponding to the defect class.
        """
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        # Open the image and convert it to RGB format
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label
 
# Set up the data directory
data_dir = "data/training_sessions/BEOL_CMP/images"
 
# Define transformations to resize all images to a consistent size
transform = transforms.Compose([
    transforms.Resize((960, 1440)),  # Resize to height=960, width=1440
    transforms.ToTensor()            # Convert images to tensors
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
        """
        Args:
            pretrained_model (nn.Module): Pretrained backbone model (e.g., ConViT).
        """
        super(DefectClassifier, self).__init__()

        # Branch 1: Binary classification (defect vs no defect)
        self.branch1 = nn.Sequential(
            pretrained_model,
            nn.Linear(1000, 2)  # Output: 2 classes
        )

        # Branch 2: Multi-class classification (23 defect labels)
        self.branch2 = nn.Sequential(
            pretrained_model,
            nn.Linear(1000, 23)  # Output: 23 classes
        )
 
    def forward(self, x):
        """
        Forward pass to handle both 6-segment and 7-segment images.
        Args:
            x: Input tensor [batch_size, channels, height, width]
        Returns:
            out1: Branch 1 output (binary classification)
            out2: Branch 2 output (multi-class classification)
        """
        def resize_segment(segment):
            # Resize the segment to 224x224 for input to the pretrained model
            return F.interpolate(segment, size=(224, 224), mode='bilinear', align_corners=False)
 
        _, _, height, width = x.shape  # Get input dimensions
 
        if height == 960 and width == 1440:  # Handle 6-segment images
            first_4_segments = resize_segment(x[:, :, :960, :960])      # Top-left 960x960
            last_2_segments = resize_segment(x[:, :, :960, 960:1440])   # Top-right 960x480
            out1 = self.branch1(first_4_segments)  # Branch 1: Defect detection
            out2 = self.branch2(last_2_segments)  # Branch 2: Defect classification
            return out1, out2
 
        elif height == 1440 and width == 1440:  # Handle 7-segment images
            optical_segment = resize_segment(x[:, :, 960:1440, :480])   # 7th segment (bottom-left)
            out1 = torch.zeros(x.size(0), 2).to(x.device)  # No defect detected
            out2 = torch.full((x.size(0), 23), -float('inf')).to(x.device)
            out2[:, 0] = 0  # SEM_NON_VISIBLE class
            return out1, out2
 
        else:
            raise ValueError(f"Unexpected image dimensions: {height}x{width}")
 
# Load pretrained ConViT model
pretrained_model = timm.create_model('convit_base', pretrained=False, num_classes=1000)
state_dict = torch.load('./convit_base.fb_in1k/pytorch_model.bin', map_location=torch.device('cpu'))
pretrained_model.load_state_dict(state_dict)
 
# Initialize the defect classifier model
model = DefectClassifier(pretrained_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
 
# Define loss function and optimizer
criterion1 = nn.BCEWithLogitsLoss()
criterion2 = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
 
# Training loop
num_epochs = 80
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
        
        defect_labels = (labels != 0).long()  # Binary classification labels
        defect_type_labels = labels  # Multi-class labels
        
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
 
    # Calculate and display training metrics
    avg_loss = total_loss / len(train_loader)
    accuracy_branch1 = correct_branch1 / total_samples * 100
    accuracy_branch2 = correct_branch2 / total_samples * 100
    print(f"Epoch [{epoch+1}/10] Training - Loss: {avg_loss:.4f}, Branch 1 Acc: {accuracy_branch1:.2f}%, Branch 2 Acc: {accuracy_branch2:.2f}%")
 
    # Validation phase
    model.eval()
    val_loss, val_correct_branch1, val_correct_branch2, val_samples = 0, 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            val_samples += labels.size(0)
 
            # Forward pass
            out1, out2 = model(images)
            defect_labels = (labels != 0).long()
            defect_type_labels = labels
            
            defect_labels = F.one_hot(defect_labels, num_classes=2).float()
 
            # Compute losses
            loss1 = criterion1(out1, defect_labels)
            loss2 = criterion2(out2, defect_type_labels)
            val_loss += loss1.item() + loss2.item()
            _, pred1 = torch.max(out1, 1)
            val_correct_branch1 += (pred1 == defect_labels.argmax(dim=1)).sum().item()
            _, pred2 = torch.max(out2, 1)
            val_correct_branch2 += (pred2 == defect_type_labels).sum().item()
 
    # Calculate and display validation metrics
    val_avg_loss = val_loss / len(val_loader)
    val_accuracy_branch1 = val_correct_branch1 / val_samples * 100
    val_accuracy_branch2 = val_correct_branch2 / val_samples * 100
    print(f"Epoch [{epoch+1}/10] Validation - Loss: {val_avg_loss:.4f}, Branch 1 Acc: {val_accuracy_branch1:.2f}%, Branch 2 Acc: {val_accuracy_branch2:.2f}%")
 
# Test phase (run after training)
model.eval()
test_loss, test_correct_branch1, test_correct_branch2, test_samples = 0, 0, 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        test_samples += labels.size(0)
 
        # Forward pass
        out1, out2 = model(images)
        defect_labels = (labels != 0).long()
        defect_type_labels = labels
        
        defect_labels = F.one_hot(defect_labels, num_classes=2).float()
 
        # Compute losses
        loss1 = criterion1(out1, defect_labels)
        loss2 = criterion2(out2, defect_type_labels)
        test_loss += loss1.item() + loss2.item()
        _, pred1 = torch.max(out1, 1)
        test_correct_branch1 += (pred1 == defect_labels.argmax(dim=1)).sum().item()
        _, pred2 = torch.max(out2, 1)
        test_correct_branch2 += (pred2 == defect_type_labels).sum().item()
 
# Calculate and display test metrics
test_avg_loss = test_loss / len(test_loader)
test_accuracy_branch1 = test_correct_branch1 / test_samples * 100
test_accuracy_branch2 = test_correct_branch2 / test_samples * 100
print(f"Test Results - Loss: {test_avg_loss:.4f}, Branch 1 Acc: {test_accuracy_branch1:.2f}%, Branch 2 Acc: {test_accuracy_branch2:.2f}%")

# Add accuracy calculation and confusion matrix for the entire training dataset

# Define function to calculate accuracy and confusion matrix for a DataLoader

def evaluate_model_on_dataset(model, dataloader, device):

    model.eval()

    all_labels = []

    all_preds_branch1 = []

    all_preds_branch2 = []
 
    with torch.no_grad():

        for images, labels in dataloader:

            images, labels = images.to(device), labels.to(device)

            out1, out2 = model(images)
 
            defect_labels = (labels != 0).long()

            defect_labels = F.one_hot(defect_labels, num_classes=2).float()
 
            _, pred_branch1 = torch.max(out1, 1)

            _, pred_branch2 = torch.max(out2, 1)
 
            all_labels.extend(labels.cpu().numpy())

            all_preds_branch1.extend(pred_branch1.cpu().numpy())

            all_preds_branch2.extend(pred_branch2.cpu().numpy())
 
    return all_labels, all_preds_branch1, all_preds_branch2
 
# Evaluate on the training dataset

train_labels, train_preds_branch1, train_preds_branch2 = evaluate_model_on_dataset(model, train_loader, device)
 
# Calculate accuracy for the entire training dataset

train_accuracy_branch1 = np.mean(np.array(train_preds_branch1) == np.array((np.array(train_labels) != 0).astype(int))) * 100

train_accuracy_branch2 = np.mean(np.array(train_preds_branch2) == np.array(train_labels)) * 100
 
print(f"Training Accuracy - Branch 1 (Binary): {train_accuracy_branch1:.2f}%, Branch 2 (Multi-class): {train_accuracy_branch2:.2f}%")
 
# Generate confusion matrix for branch 2 (multi-class classification)

cm = confusion_matrix(train_labels, train_preds_branch2, labels=list(range(23)))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)

disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')

plt.title("Confusion Matrix - Training Set")

plt.savefig("confusion_matrix.png")

plt.show()
 
# Visualize heatmaps for several misclassified images

def visualize_heatmaps(model, dataloader, device):

    model.eval()

    misclassified_images = []

    misclassified_preds = []

    misclassified_labels = []
 
    # Identify misclassified samples

    with torch.no_grad():

        for images, labels in dataloader:

            images, labels = images.to(device), labels.to(device)

            out1, out2 = model(images)
 
            _, pred_branch2 = torch.max(out2, 1)
 
            for i in range(len(labels)):

                if pred_branch2[i] != labels[i]:

                    misclassified_images.append(images[i].cpu())

                    misclassified_preds.append(pred_branch2[i].item())

                    misclassified_labels.append(labels[i].item())
 
    # Visualize heatmaps for the first few misclassified samples

    for i in range(min(5, len(misclassified_images))):

        img = misclassified_images[i].permute(1, 2, 0).numpy()

        plt.imshow(img)

        plt.title(f"True: {misclassified_labels[i]}, Pred: {misclassified_preds[i]}")

        plt.axis("off")

        plt.savefig(f"misclassified_image_{i}.png")

        plt.show()
 
# Visualize heatmaps for misclassified training samples

visualize_heatmaps(model, train_loader, device)

# Save the trained model
torch.save(model.state_dict(), "model_new.pth")
print("Model saved to 'model_new.pth'.")
 
