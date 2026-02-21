import torch
import torch.nn as nn
import torchvision.models as models


def build_model(num_classes=3, device="cuda"):
    """
    Build ResNet50 model and replace final layer
    """
    model = models.resnet50(pretrained=True)

    # Replace final FC layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)
    return model


def freeze_backbone(model):
    """
    Freeze all layers except final FC
    """
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_last_block(model):
    """
    Unfreeze last residual block (layer4) for fine-tuning
    """
    for param in model.layer4.parameters():
        param.requires_grad = True

    return model


def get_optimizer(model, lr):
    """
    Return optimizer only for trainable params
    """
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )


def get_loss_function():
    return nn.CrossEntropyLoss()



def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc.item()


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)

    val_loss = running_loss / len(dataloader)
    val_acc = correct.double() / len(dataloader.dataset)

    return val_loss, val_acc.item()


def save_model(model, path="resnet50_model.pth"):
    torch.save(model.state_dict(), path)
    
print ("Model architecture and training functions defined successfully!")   