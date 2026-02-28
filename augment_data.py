import os
import torch
from PIL import Image
from torchvision import transforms

# --- Configuration ---
INPUT_FOLDER = "fruit-dataset-before-augmentation/"     # Your folder of original images
OUTPUT_FOLDER = "fruit-dataset-after-augmentation/"   # Where the new modified images will be saved
COPIES_PER_IMAGE = 2                  # How many augmented versions to make per original

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Torchvision doesn't have a simple built-in Gaussian noise filter, so we make a quick custom one
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.) # Keep pixel values valid

# --- Define the Augmentation Pipeline ---
# The 'p' value is the probability (0.0 to 1.0) that the effect gets applied to an image.
augmentations = transforms.Compose([
    # 1. Rotation: Randomly rotates between -90 and 90 degrees
    transforms.RandomApply([transforms.RandomRotation(degrees=(-90, 90))], p=0.5),
    
    # 2 & 3. Translation & Scaling: Shifts up to 20% off-center and scales up to 110%
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(1.0, 1.1))], p=0.5),
    
    # 4. Flipping: 50% chance to flip horizontally
    transforms.RandomHorizontalFlip(p=0.5),
    
    # 5. Noise: Convert to PyTorch tensor, apply Gaussian noise, convert back to image
    transforms.ToTensor(),
    transforms.RandomApply([AddGaussianNoise(std=0.05)], p=0.3),
    transforms.ToPILImage()
])

# --- Processing Loop ---
valid_extensions = (".jpg", ".jpeg", ".png")
print("Starting data augmentation...")

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(valid_extensions):
        image_path = os.path.join(INPUT_FOLDER, filename)
        
        try:
            # Open original image
            original_image = Image.open(image_path).convert('RGB')
            
            # Generate the requested number of augmented copies
            for i in range(COPIES_PER_IMAGE):
                aug_image = augmentations(original_image)
                
                # Create a new filename (e.g., aug_1_apples_42.jpg)
                new_filename = f"aug_{i+1}_{filename}"
                save_path = os.path.join(OUTPUT_FOLDER, new_filename)
                
                aug_image.save(save_path)
                
            print(f"Augmented: {filename} -> Created {COPIES_PER_IMAGE} new versions.")
            
        except Exception as e:
            print(f"Could not process {filename}: {e}")

print(f"\nDone! Augmented images saved to the '{OUTPUT_FOLDER}' directory.")
