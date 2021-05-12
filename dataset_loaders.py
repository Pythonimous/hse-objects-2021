from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import pickle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class YelpDataset(Dataset):
    """Is It A Restaurant Dataset."""

    def __init__(self, csv_file, photos_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_names = csv_file.iloc[:, 0]
        self.labels = csv_file.iloc[:, 1:].to_numpy(dtype='float')
        self.photos_dir = photos_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.photos_dir,
                                self.image_names.iloc[idx] + '.jpg')
        image = Image.open(img_name)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, labels


class ObjDataset(Dataset):

    def __init__(self, csv_file, trf_features, object_features):

        self.image_names = csv_file.iloc[:, 0]
        self.labels = csv_file.iloc[:, 1:].to_numpy(dtype='float')

        self.object_features = object_features
        self.trf_features = trf_features

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        labels = torch.tensor(self.labels[idx], dtype=torch.float32)

        confidence_vector = np.zeros(91)
        counts_vector = np.zeros(91)
        for box in self.object_features[self.image_names.iloc[idx]]:
            if box:
                _, i, confidence = box
                confidence_vector[i] = max(confidence_vector[i], confidence)
                counts_vector[i] += 1
        obj_feature_vector = np.concatenate((confidence_vector, counts_vector))
        trf_feature_vector = self.trf_features[self.image_names.iloc[idx]]
        obj_norm = np.linalg.norm(obj_feature_vector)
        obj_feature_vector /= obj_norm
        feature_vector = np.concatenate((trf_feature_vector, obj_feature_vector))
        feature_tensor = torch.from_numpy(feature_vector).to(torch.float32)

        return feature_tensor, labels


def build_datasets(flags, data_frames, transforms):
    """
    data_frames: dict of {train, dev, test} frames to be used for data loaders
    photos_dir: directory where photos are;
    transforms: dict of {train, dev, test} transformations.
    """
    image_datasets = {x: YelpDataset(data_frames[x], flags['photo_dir'], transforms[x]) for x in ['train', 'dev', 'test']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: DataLoader(image_datasets[x],
                                      batch_size=flags['batch_size'],
                                      shuffle=True,
                                      num_workers=flags['num_workers'],
                                      pin_memory=True) for x in ['train', 'dev']}
    dataloaders_dict['test'] = DataLoader(image_datasets['test'],
                                          batch_size=flags['batch_size'],
                                          shuffle=False,
                                          num_workers=flags['num_workers'],
                                          pin_memory=True)
    return dataloaders_dict
