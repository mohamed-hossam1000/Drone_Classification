from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np

transform = transforms.Compose([transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])

def load_data(data_dir,transform):
    # Loading the dataset and applying transforms
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    return dataset


def split_dataloader(data,train_split):
    train_size = int(train_split * len(data))
    test_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, test_size])  
    trainL = DataLoader(train_data, batch_size=64, shuffle=True)
    valL = DataLoader(val_data, batch_size=64, shuffle=False)
    
    return trainL,valL


def preprocess_image(image_np,transform):

    # Convert numpy array to PIL Image
    if isinstance(image_np, np.ndarray):
        image = Image.fromarray(image_np.astype('uint8'))
    else:
        raise TypeError("Input must be a numpy array")

    image = transform(image)

    # Add batch dimension → (1, C, H, W)
    image = image.unsqueeze(0)

    return image
