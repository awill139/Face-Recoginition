import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Define data transforms


def get_data(split='train'):
    if split=='train':
        folder = 'new_att'
    else:
        folder = 'test_att'

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    data = datasets.ImageFolder(folder, transform=transform)
    loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    return loader