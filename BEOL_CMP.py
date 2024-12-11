import os

import torch

import torch.nn as nn

from transformers import AutoModelForImageClassification

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from PIL import Image
 
# Path constants

DATA_DIR = "data/training_sessions/BEOL_CMP/images"

PRETRAINED_MODEL_DIR = "MahmoodLab_UNI"  # Path to your locally downloaded model directory
 
# Device configuration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Dataset class

class DefectDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir

        self.transform = transform

        self.image_paths = []

        self.labels = []

        self._prepare_data()
 
    def _prepare_data(self):

        for label, folder in enumerate(os.listdir(self.root_dir)):

            folder_path = os.path.join(self.root_dir, folder)

            if os.path.isdir(folder_path):

                for image_file in os.listdir(folder_path):

                    image_path = os.path.join(folder_path, image_file)

                    self.image_paths.append(image_path)

                    self.labels.append(label)
 
    def __len__(self):

        return len(self.image_paths)
 
    def __getitem__(self, idx):

        image_path = self.image_paths[idx]

        label = self.labels[idx]

        image = Image.open(image_path).convert("L")  # Convert to grayscale
 
        # Split image into 6 segments

        image = transforms.ToTensor()(image)

        top_left = image[:, :480, :480]

        top_middle = image[:, :480, 480:960]

        top_right = image[:, :480, 960:]

        bottom_left = image[:, 480:, :480]

        bottom_middle = image[:, 480:, 480:960]

        bottom_right = image[:, 480:, 960:]
 
        # For branch 1: first 4 segments

        branch1_input = torch.cat((top_left, top_middle, bottom_left, bottom_middle), dim=0)

        # For branch 2: last 2 segments

        branch2_input = torch.cat((top_right, bottom_right), dim=0)
 
        if self.transform:

            branch1_input = self.transform(branch1_input)

            branch2_input = self.transform(branch2_input)
 
        return branch1_input, branch2_input, label
 
# Define the model

class DefectClassifier(nn.Module):

    def __init__(self, pretrained_model_dir):

        super(DefectClassifier, self).__init__()
 
        # Load the pretrained model from local directory

        self.branch1 = AutoModelForImageClassification.from_pretrained(pretrained_model_dir, num_labels=1)  # Defect presence detection

        self.branch2 = AutoModelForImageClassification.from_pretrained(pretrained_model_dir, num_labels=23)  # Defect type classification
 
        # Final classifier to combine both branches' outputs

        self.final_classifier = nn.Linear(24, 23)
 
    def forward(self, branch1_input, branch2_input):

        # Extract features for both branches

        branch1_features = self.branch1(branch1_input).logits

        branch2_features = self.branch2(branch2_input).logits
 
        # Concatenate outputs

        combined_output = torch.cat((branch1_features, branch2_features), dim=1)
 
        # Final output from combined branches

        final_output = self.final_classifier(combined_output)
 
        return final_output
 
# Main function

def main():

    # Transforms for data preprocessing

    transform = transforms.Compose([

        transforms.Resize((224, 224)),  # Resize for pretrained model input

        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale

    ])
 
    # Prepare dataset and dataloaders

    dataset = DefectDataset(DATA_DIR, transform=transform)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
 
    # Initialize model

    model = DefectClassifier(PRETRAINED_MODEL_DIR)

    model.to(device)
 
    # Loss and optimizer

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
    # Training loop

    num_epochs = 10

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0
 
        for branch1_input, branch2_input, labels in dataloader:

            branch1_input, branch2_input, labels = branch1_input.to(device), branch2_input.to(device), labels.to(device)
 
            # Forward pass

            outputs = model(branch1_input, branch2_input)
 
            loss = criterion(outputs, labels)
 
            # Backward pass

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
 
            running_loss += loss.item()
 
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
 
    print("Training complete.")
 
if __name__ == "__main__":

    main()

 