from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
from torchvision import models, transforms
from PIL import Image

import time
import os
from tqdm import tqdm

torch.set_num_threads(4)
photo_dir = './yelp_photos/photos/'
csv_dir = "./yelp_data/business_restaurant"

# Define Parameters
FLAGS = dict()
FLAGS['batch_size'] = 4
FLAGS['num_workers'] = 0
FLAGS['feature_extract'] = True


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, feature_extract, use_pretrained=True):
    
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        modules = list(model_ft.children())[:-1]
        model_ft = nn.Sequential(*modules)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        new_classifier = nn.Sequential(*list(model_ft.classifier.children())[:-2])
        model_ft.classifier = new_classifier
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier = model_ft.classifier[:-3]
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Identity()
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


img_transforms = transforms.Compose([
                 transforms.Resize(224),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class IsRestaurantDataset(Dataset):
    """Is It A Restaurant Dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.img_names = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_names[idx].split('.')[0]
        img_path = os.path.join(self.root_dir,
                                img_name + '.jpg')
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, img_name


trans_dataset = IsRestaurantDataset(photo_dir, img_transforms)
# Create training and validation dataloaders
trans_loader = torch.utils.data.DataLoader(trans_dataset, batch_size=FLAGS['batch_size'], shuffle=False, num_workers=FLAGS['num_workers'])


def extract_trf_features(model_name):
    model_ft, input_size = initialize_model(model_name, FLAGS['feature_extract'], use_pretrained=True)

    feat_dict = {}
    with torch.no_grad():
        for num, data in enumerate(tqdm(trans_loader)):
            images, img_names = data

            vectors = model_ft(images)
            for idx in range(len(vectors)):
                vector = vectors[idx].numpy()
                feat_dict[img_names[idx]] = vector
            if not num+1 % 250:
                time.sleep(10)

    np.savez(f'yelp_data/transfer_features/{model_name}_features.npz', feat_dict)


for model_name in ['alexnet', 'vgg', 'resnet', 'densenet']:
    extract_trf_features(model_name)
