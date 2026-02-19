import os
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# 1. Setup your paths (Change this to your actual folder name!)
IMAGE_FOLDER = "FRUIT-CAPTION/data/DONNEE_ORIGINAL_AVANT_TRANSFORMATION/"
OUTPUT_JSON = "dataset_captions.json"

# 2. Load the BLIP model from Hugging Face
print("Loading the AI model (this might take a minute)...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# If you have a GPU, this sends the model there to run much faster
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 3. Process the images
dataset_results = []
valid_extensions = (".jpg", ".jpeg", ".png") # Add others if needed

print(f"Starting caption generation using {device}...")

# Loop through every file in your folder
for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith(valid_extensions):
        image_path = os.path.join(IMAGE_FOLDER, filename)

        # Open the image
        raw_image = Image.open(image_path).convert('RGB')

        # Translate the image into data the model can read
        inputs = processor(raw_image, return_tensors="pt").to(device)

        # Generate the text caption
        out = model.generate(**inputs, max_new_tokens=20)
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Save the pair to our list
        dataset_results.append({
            "image": filename,
            "caption": caption
        })
        print(f"Success -> {filename}: {caption}")

# 4. Save everything to a JSON file
with open(OUTPUT_JSON, "w") as f:
    json.dump(dataset_results, f, indent=4)

print(f"\nDone! Saved {len(dataset_results)} captions to {OUTPUT_JSON}.")
