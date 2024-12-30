import os
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import cv2
from torchvision.transforms.functional import normalize
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
 
# Define the defect classifier model
class DefectClassifier(nn.Module):
    def __init__(self, pretrained_model):
        super(DefectClassifier, self).__init__()
        self.branch1 = nn.Sequential(
            pretrained_model,
            nn.Linear(1000, 2)  # Binary classification: defect or no defect
        )

        self.branch2 = nn.Sequential(
            pretrained_model,
            nn.Linear(1000, 23)  # Multi-class classification: 23 defect types
        )
 
    def forward(self, x):
        def resize_segment(segment):
            return nn.functional.interpolate(segment, size=(224, 224), mode='bilinear', align_corners=False)
 
        _, _, height, width = x.shape
 
        if height == 960 and width == 1440:
            # Split and resize for branches
            first_4_segments = resize_segment(x[:, :, :960, :960])
            last_2_segments = resize_segment(x[:, :, :960, 960:])
            out1 = self.branch1(first_4_segments)
            out2 = self.branch2(last_2_segments)
            return out1, out2
 
        elif height == 1440 and width == 1440:  # Handle 7-segment images
            optical_segment = resize_segment(x[:, :, 960:1440, :480])  # 7th segment (bottom-left)
            out1 = torch.zeros(x.size(0), 2).to(x.device)  # No defect detected
            out2 = torch.full((x.size(0), 23), -float('inf')).to(x.device)
            out2[:, 0] = 0  # SEM_NON_VISIBLE class
            return out1, out2
 
        else:
            raise ValueError(f"Unexpected image dimensions: {height}x{width}")
 
# Load the pretrained ConViT model
pretrained_model = timm.create_model('convit_base', pretrained=False, num_classes=1000)
state_dict = torch.load('./convit_base.fb_in1k/pytorch_model.bin', map_location=torch.device('cpu'))
pretrained_model.load_state_dict(state_dict)

# Initialize and load the trained model
model = DefectClassifier(pretrained_model)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()
 
# Define transformations for the unseen dataset
transform = transforms.Compose([
    transforms.Resize((960, 1440)),
    transforms.ToTensor()
])
 
# Define the training defect labels (to filter unseen data)
training_defect_labels = sorted([
    'BALL_DEFECTS', 'BIG_PARTICLES', 'C_RESIDUE',  # Add all 23 training labels here
    'CROWN', 'CU_BRIDGING', 'CU_CRATER',
    'CU_DISHING', 'CU_HILLOCK', 'CU_MISSING',
    'CU_PUDDLE', 'CU_SURFACE_EROSION', 'DEFOCUS',
    'EBR_SPLASH_BACK', 'FILM_BUMP', 'FLAKE',
    'KILLER_SCRATCH', 'LINER_VOIDS', 'OXIDE_CRATER',
    'POLISH_SCRATCH', 'RESIDUE', 'SEM_NON-VISIBLE',
    'SINGULAR_CU_VOID', 'SMALL_PARTICLES'
])

label_to_index = {label: idx for idx, label in enumerate(training_defect_labels)}
 
# Load unseen dataset (organized by folders representing labels)
class UnseenDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []
 
        # Traverse directory structure
        for defect_type in os.listdir(root_dir):
            if defect_type not in label_to_index:
                continue  # Skip labels not in the training set
              
            defect_dir = os.path.join(root_dir, defect_type)
            if os.path.isdir(defect_dir):
                for file in os.listdir(defect_dir):
                    if file.endswith(('.jpg', '.png', '.jpeg')):
                        self.file_paths.append(os.path.join(defect_dir, file))
                        self.labels.append(label_to_index[defect_type])
 
    def __len__(self):
        return len(self.file_paths)
 
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
 
unseen_data_dir = "data/training_sessions/BEOL_CMP_test/images"
unseen_dataset = UnseenDataset(unseen_data_dir, transform=transform)
unseen_loader = DataLoader(unseen_dataset, batch_size=16, shuffle=False)
 
# Evaluate the model on the unseen dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
 
criterion = nn.CrossEntropyLoss()

unseen_loss, unseen_correct_branch1, unseen_correct_branch2, unseen_samples = 0, 0, 0, 0
 
misclassified_images = []
misclassified_labels = []
all_labels = []
all_preds = []
 
with torch.no_grad():
    for images, labels in unseen_loader:
        images, labels = images.to(device), labels.to(device)
        unseen_samples += labels.size(0)
 
        try:
            out1, out2 = model(images)
 
            # Branch 1: Binary classification (defect or no defect)
            defect_labels = (labels != 0).long()  # Convert multi-class to binary labels
            loss1 = criterion(out1, defect_labels)
            _, pred1 = torch.max(out1, 1)
            unseen_correct_branch1 += (pred1 == defect_labels).sum().item()
 
            # Branch 2: Multi-class classification
            loss2 = criterion(out2, labels)
            _, pred2 = torch.max(out2, 1)
            unseen_correct_branch2 += (pred2 == labels).sum().item()
 
            # Total loss
            unseen_loss += loss1.item() + loss2.item()
 
            # Collect misclassified samples
            misclassified = (pred2 != labels)
            misclassified_images.extend(images[misclassified].cpu())
            misclassified_labels.extend(labels[misclassified].cpu())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred2.cpu().numpy())
 
        except Exception as e:
            print(f"Error during evaluation: {e}")
 
unseen_avg_loss = unseen_loss / len(unseen_loader)
unseen_accuracy_branch1 = unseen_correct_branch1 / unseen_samples * 100
unseen_accuracy_branch2 = unseen_correct_branch2 / unseen_samples * 100
 
print(f"Unseen Data Results - Loss: {unseen_avg_loss:.4f}, "
      f"Branch 1 Acc: {unseen_accuracy_branch1:.2f}%, "
      f"Branch 2 Acc: {unseen_accuracy_branch2:.2f}%")
 
# Function to generate Grad-CAM heatmaps
def generate_gradcam_heatmap(model, layer_name, images, labels, device, save_dir="heatmaps"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
 
    model.eval()
 
    for idx, (image, label) in enumerate(zip(images, labels)):
        image = image.unsqueeze(0).to(device)  # Add batch dimension

        def forward_hook(module, input, output):
            nonlocal feature_maps
            feature_maps = output
 
        def backward_hook(module, grad_input, grad_output):
            nonlocal grads
            grads = grad_output[0]
 
        feature_maps, grads = None, None
        layer = dict(model.named_modules())[layer_name]
 
        # Attach hooks
        forward_handle = layer.register_forward_hook(forward_hook)
        backward_handle = layer.register_backward_hook(backward_hook)
 
        # Forward and backward pass
        outputs = model(image)
        branch1_output, branch2_output = outputs
        branch2_loss = criterion(branch2_output, label.unsqueeze(0))  # Use multi-class output
        branch2_loss.backward()
 
        # Compute Grad-CAM
        weights = grads.mean(dim=(2, 3), keepdim=True)  # Average gradients over spatial dimensions
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam).squeeze().cpu().numpy()
 
        # Normalize CAM to [0, 1] and resize
        cam = cv2.resize(cam, (1440, 960))  # Resize to match original image dimensions
        cam = (cam - cam.min()) / (cam.max() - cam.min())
 
        # Overlay heatmap on image
        original_image = image.cpu().squeeze().permute(1, 2, 0).numpy()
        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.uint8(255 * original_image), 0.5, heatmap, 0.5, 0)
 
        # Save heatmap
        label_name = list(label_to_index.keys())[list(label_to_index.values()).index(label.item())]
        save_path = os.path.join(save_dir, f"image_{idx}_label_{label_name}.jpg")
        cv2.imwrite(save_path, overlay)
        print(f"Heatmap saved at {save_path}")
 
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()
 
# Generate heatmaps for misclassified images using Grad-CAM
layer_to_visualize = "branch2.0.blocks.2.mlp.fc2"  # Example layer name (change if necessary)
generate_gradcam_heatmap(model, layer_to_visualize, misclassified_images, misclassified_labels, device)
 
# Generate and save confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(label_to_index.values()))
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=training_defect_labels, yticklabels=training_defect_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for Defect Classification")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
 
confusion_matrix_path = "confusion_matrix.png"
plt.savefig(confusion_matrix_path)
plt.close()
print(f"Confusion matrix saved at {confusion_matrix_path}")
 
# Print classification report
report = classification_report(all_labels, all_preds, target_names=training_defect_labels)
print("Classification Report:\n")
print(report)
 
