import os

import torch

import timm

import torch.nn as nn

import torch.nn.functional as F

from torchvision import datasets, transforms

from torch.utils.data import DataLoader

from PIL import Image
 
# Define dataset class to handle segmentation

class DefectDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir

        self.transform = transform

        self.classes = os.listdir(root_dir)

        self.file_paths = []

        self.labels = []
 
        for label, defect_class in enumerate(self.classes):

            defect_dir = os.path.join(root_dir, defect_class)

            for img_file in os.listdir(defect_dir):

                self.file_paths.append(os.path.join(defect_dir, img_file))

                self.labels.append(label)
 
    def __len__(self):

        return len(self.file_paths)
 
    def __getitem__(self, idx):

        img_path = self.file_paths[idx]

        label = self.labels[idx]

        image = Image.open(img_path).convert('L')  # Load greyscale image

        if self.transform:

            image = self.transform(image)

        return image, label
 
# Data directory

data_dir = "data/training_sessions/BEOL_CMP/images"
 
# Define transformations to resize all images to a consistent size

transform = transforms.Compose([

    transforms.Resize((1440, 1440)),  # Resize to 1440x1440

    transforms.ToTensor()

])
 
# Create dataset and dataloader

dataset = DefectDataset(data_dir, transform=transform)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
 
# Define the model with two branches

class DefectClassifier(nn.Module):

    def __init__(self, pretrained_model):

        super(DefectClassifier, self).__init__()

        self.branch1 = nn.Sequential(

            pretrained_model,

            nn.Linear(768, 2)  # Binary classification: defect or no defect

        )

        self.branch2 = nn.Sequential(

            pretrained_model,

            nn.Linear(768, 23)  # 23 defect labels

        )

        self.optical_classifier = nn.Linear(768, 2)  # Optical: visible or non-visible
 
    def forward(self, x):

        def resize_segment(segment):

            return F.interpolate(segment, size=(224, 224), mode='bilinear', align_corners=False)
 
        # Split input into segments

        first_4_segments = resize_segment(x[:, :, :960, :960])  # [N, C, 224, 224]

        last_2_segments = resize_segment(x[:, :, :960, 960:1440])
 
        # Branch 1: Detect defects

        out1 = self.branch1(first_4_segments)
 
        # Branch 2: Classify defect type

        out2 = self.branch2(last_2_segments)
 
        # Combine outputs

        if out1.argmax(1) == 0 and out2.argmax(1) == 0:  # No defect

            optical_segment = resize_segment(x[:, :, 960:1440, :480])

            optical_output = self.optical_classifier(optical_segment)

            return optical_output
 
        return out1, out2
 
# Load pretrained ConViT model

pretrained_model = timm.create_model('convit_base', pretrained=False)

state_dict = torch.load('./convit_base.fb_in1k/pytorch_model.bin', map_location=torch.device('cpu'))

pretrained_model.load_state_dict(state_dict)
 
# Initialize defect classifier model

model = DefectClassifier(pretrained_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
 
# Define loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
 
# Training loop

for epoch in range(10):  # Example: 10 epochs

    model.train()

    total_loss = 0
 
    for images, labels in dataloader:

        images, labels = images.to(device), labels.to(device)
 
        # Forward pass

        outputs = model(images)
 
        if isinstance(outputs, tuple):

            out1, out2 = outputs

            loss1 = criterion(out1, labels)

            loss2 = criterion(out2, labels)

            loss = loss1 + loss2

        else:

            loss = criterion(outputs, labels)
 
        # Backward pass

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
 
        total_loss += loss.item()
 
    print(f"Epoch [{epoch+1}/10], Loss: {total_loss:.4f}")
 
print("Training complete!")

 
