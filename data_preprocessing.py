from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, random_split

transform = transforms.Compose([transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])

def load_data(data_dir):
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

def preprocess_image(image):
    preprocessed_image = transform(image)

    return preprocess_image