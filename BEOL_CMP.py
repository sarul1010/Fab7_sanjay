import os
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
 
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
 
# Create dataset and DataLoader for batching
dataset = DefectDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
 
 
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
 
        # Handle 6-segment images (960x1440)
        if height == 960 and width == 1440:
            # Split input into first 4 and last 2 segments
            first_4_segments = resize_segment(x[:, :, :960, :960])      # Top-left 960x960
            last_2_segments = resize_segment(x[:, :, :960, 960:1440])   # Top-right 960x480
 
            # Branch 1: Detect defects
            out1 = self.branch1(first_4_segments)
 
            # Branch 2: Classify defect type
            out2 = self.branch2(last_2_segments)
 
            return out1, out2
 
        # Handle 7-segment images (1440x1440)
        elif height == 1440 and width == 1440:
            # Extract relevant segments
            first_4_segments = resize_segment(x[:, :, :960, :960])      # Top-left 960x960
            last_2_segments = resize_segment(x[:, :, :960, 960:1440])   # Top-right 960x480
            optical_segment = resize_segment(x[:, :, 960:1440, :480])   # 7th segment (bottom-left 480x480)
 
            # For 7-segment images, classify as SEM_NON_VISIBLE
            out1 = torch.zeros(x.size(0), 2).to(x.device)  # No defect detected
            out2 = torch.full((x.size(0), 23), -float('inf')).to(x.device)  # All logits negative
            out2[:, 0] = 0  # Set logit for SEM_NON_VISIBLE (class 0) to 0
 
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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
 
# Training loop
for epoch in range(10):  # Training for 10 epochs
    model.train()
    total_loss = 0
    correct_branch1 = 0  # Correct predictions for Branch 1 (binary)
    correct_branch2 = 0  # Correct predictions for Branch 2 (multi-class)
    total_samples = 0
 
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        batch_size = labels.size(0)  # Get batch size
        total_samples += batch_size
 
        # Forward pass
        out1, out2 = model(images)
 
        # Prepare branch-specific labels
        defect_labels = (labels != 0).long()  # Binary: defect or no defect (0=No Defect, 1=Defect)
        defect_type_labels = labels           # Multi-class: defect type
 
        # Compute losses for both branches
        loss1 = criterion(out1, defect_labels)  # Loss for binary classification
        loss2 = criterion(out2, defect_type_labels)  # Loss for multi-class classification
        loss = loss1 + loss2  # Total loss
 
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
 
        # Compute predictions and accuracy for Branch 1 (binary classification)
        _, pred1 = torch.max(out1, 1)  # Get predictions from out1
        correct_branch1 += (pred1 == defect_labels).sum().item()
 
        # Compute predictions and accuracy for Branch 2 (multi-class classification)
        _, pred2 = torch.max(out2, 1)  # Get predictions from out2
        correct_branch2 += (pred2 == defect_type_labels).sum().item()
 
    # Calculate average loss and accuracy after the epoch
    avg_loss = total_loss / len(dataloader)
    accuracy_branch1 = correct_branch1 / total_samples * 100  # Accuracy for Branch 1
    accuracy_branch2 = correct_branch2 / total_samples * 100  # Accuracy for Branch 2
 
    # Print metrics for the epoch
    print(f"Epoch [{epoch+1}/10]")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Branch 1 Accuracy (Defect/No Defect): {accuracy_branch1:.2f}%")
    print(f"  Branch 2 Accuracy (Defect Type): {accuracy_branch2:.2f}%\n")
 
print("Training complete!")
