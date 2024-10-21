import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import SnoutNet
from datasetLoader import test_loader
from train import EuclideanDistanceLoss

def load_model(model_path, device):
    model = SnoutNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def calculate_statistics(distances):
    min_distance = np.min(distances)
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    std_distance = np.std(distances)
    return min_distance, mean_distance, max_distance, std_distance

def test_model(model, test_loader, device):
    distances = []
    images = []
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            distances_batch = torch.sqrt(torch.sum((outputs - labels) ** 2, dim=1)).cpu().numpy()
            distances.extend(distances_batch)
            images.extend(inputs.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())
    return np.array(distances), np.array(images), np.array(predictions), np.array(ground_truths)

def denormalize(img, mean, std):
    img = img * std + mean
    return np.clip(img, 0, 1)

def display_images(images, predictions, ground_truths, num_images=5):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i in range(num_images):
        img = np.transpose(images[i], (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        img = denormalize(img, mean, std)  # Denormalize the image
        pred = predictions[i]
        gt = ground_truths[i]

        plt.imshow(img)
        plt.scatter([pred[0]], [pred[1]], c='r', label='Predicted')
        plt.scatter([gt[0]], [gt[1]], c='g', label='Ground Truth')
        plt.legend()
        plt.show()

def main():
    model_path = 'weights.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    model = load_model(model_path, device)
    distances, images, predictions, ground_truths = test_model(model, test_loader, device)
    min_distance, mean_distance, max_distance, std_distance = calculate_statistics(distances)

    print(f'Minimum Euclidean Distance: {min_distance}')
    print(f'Mean Euclidean Distance: {mean_distance}')
    print(f'Maximum Euclidean Distance: {max_distance}')
    print(f'Standard Deviation of Euclidean Distance: {std_distance}')

    display_images(images, predictions, ground_truths, num_images=5)

if __name__ == '__main__':
    main()