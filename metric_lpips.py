import os
import torch
import lpips
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Specify your folders here
hr_folder = "./DIV2K/DIV2K_valid_HR"
sr_folder = "./Output/DIV2K_X4/FSRCNN/results/SR2"

# Initialize LPIPS model with AlexNet
loss_fn = lpips.LPIPS(net='alex')
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# Transform to convert images to tensors and normalize to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Get all image files in HR folder
hr_files = [f for f in os.listdir(hr_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# Calculate LPIPS for each pair
lpips_values = []

print(f"Processing {len(hr_files)} image pairs...")
for filename in hr_files:
    sr_path = os.path.join(sr_folder, filename)

    # Skip if SR image doesn't exist
    if not os.path.exists(sr_path):
        print(f"Skipping {filename} - not found in SR folder")
        continue

    # Load and preprocess images
    hr_img = Image.open(os.path.join(hr_folder, filename)).convert('RGB')
    sr_img = Image.open(sr_path).convert('RGB')

    # Transform images to tensors
    hr_tensor = transform(hr_img).unsqueeze(0)
    sr_tensor = transform(sr_img).unsqueeze(0)

    # Move to GPU if available
    if torch.cuda.is_available():
        hr_tensor = hr_tensor.cuda()
        sr_tensor = sr_tensor.cuda()

    # Calculate LPIPS
    with torch.no_grad():
        dist = loss_fn.forward(hr_tensor, sr_tensor)

    # Store the score
    lpips_score = dist.item()
    lpips_values.append(lpips_score)
    print(f"{filename}: {lpips_score:.6f}")

# Calculate and print average
average_lpips = np.mean(lpips_values)
print(f"\nAverage LPIPS: {average_lpips:.6f}")

# Save results to file
with open("lpips_results.txt", "w") as f:
    f.write(f"Average LPIPS: {average_lpips:.6f}\n\n")
    f.write("Individual scores:\n")
    for filename, score in zip(hr_files, lpips_values):
        f.write(f"{filename}: {score:.6f}\n")

print("Results saved to lpips_results.txt")