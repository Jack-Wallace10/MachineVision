import matplotlib.pyplot as plt
from datasetLoader import train_loader

def visualize_sample(image, label):
    plt.imshow(image.permute(1, 2, 0))
    plt.scatter(label[0], label[1], c='red', s=40)
    plt.show()

# Iterate through the dataset and print values
for images, labels in train_loader:
    for i in range(len(images)):
        print(f'Image shape: {images[i].shape}, Label: {labels[i]}')
        visualize_sample(images[i], labels[i])
    break  # Remove this break to iterate through the entire dataset