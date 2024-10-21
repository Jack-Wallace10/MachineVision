import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, apply_augmentation=False):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform if transform is not None else transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor()])
        self.target_transform = target_transform
        self.apply_augmentation = apply_augmentation

        # Define augmentations
        self.augmentation_1 = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip with 50% probability
            transforms.ToTensor()
        ])
        
        self.augmentation_2 = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.RandomRotation(30),  # Rotate by random angle within 30 degrees
            transforms.ToTensor()
        ])

        self.combined_augmentation = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),  # Horizontal flip + rotation
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_filename = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_filename)
        try:
            image = Image.open(img_path).convert('RGB')
        except UnidentifiedImageError:
            print(f"Cannot identify image file {img_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))  # Skip to the next image
        
        label = eval(self.img_labels.iloc[idx, 1])  # Ground truth label as (x, y) coordinates

        # Apply augmentations if enabled
        if self.apply_augmentation:
            # Choose the combined augmentation (Horizontal Flip + Rotation)
            image = self.combined_augmentation(image)
        else:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, torch.tensor(label, dtype=torch.float32)

    # Initialize dataset and dataloader with resizing transformation
images_dir = '/Users/jackwallace/Desktop/ELEC475/Lab2/oxford-iiit-pet-noses/images-original/images'
train_labels_file = '/Users/jackwallace/Desktop/ELEC475/Lab2/oxford-iiit-pet-noses/train_noses.txt'
test_labels_file = '/Users/jackwallace/Desktop/ELEC475/Lab2/oxford-iiit-pet-noses/test_noses.txt'

batch_size = 16  # Batch size

# Define dataset with no augmentations for training (Step 4)
train_dataset_no_aug = CustomImageDataset(train_labels_file, images_dir, apply_augmentation=False)
train_loader_no_aug = DataLoader(train_dataset_no_aug, batch_size=batch_size, shuffle=True)

# Define dataset with augmentations for training (Step 5)
train_dataset_with_aug = CustomImageDataset(train_labels_file, images_dir, apply_augmentation=True)
train_loader_with_aug = DataLoader(train_dataset_with_aug, batch_size=batch_size, shuffle=True)

# Define test dataset (no augmentations for testing)
test_dataset = CustomImageDataset(test_labels_file, images_dir, apply_augmentation=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Function to visualize data for reality check
def visualize_data(loader, num_images=5):
    for i, (images, labels) in enumerate(loader):
        if i >= num_images:
            break
        img = images[0].numpy().transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        img = img.clip(0, 1)  # Clip image values to [0, 1] for display
        label = labels[0].numpy()

        print(f"Ground Truth Label: {label}")
        
        # Show image with ground truth label
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.scatter([label[0]], [label[1]], c='g', label='Ground Truth')
        plt.legend()
        plt.show()

# Visualize some test images
visualize_data(test_loader, num_images=5) 