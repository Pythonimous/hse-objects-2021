import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class YelpDataset(Dataset):
    """Generic Yelp dataset."""

    def __init__(self, csv_file, photos_dir, transform=None):
        """
        Args:
            csv_file (DataFrame): Loaded pandas dataframe: photos in col 1, classes in other cols.
            photos_dir (str): Path to the directory with the images.
            transform (callable, optional): Optional transform to be applied on a sample.
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


def build_datasets(flags, data_frames, transforms):
    """
    Initializes train, dev, test dataloaders for further usage.
    Args:
        flags (dict): config-wide flags (path to photos, batch size, etc.)
        data_frames (dict): dict of {train, dev, test} frames to be used for data loaders
        transforms: (dict): dict of {train, dev, test} transformations.
    Returns:
        dataloaders_dict (dict): dict of {train, dev, test} dataloaders
    """
    image_datasets = {x: YelpDataset(data_frames[x],
                                     flags['photo_dir'],
                                     transforms[x]) for x in ['train', 'dev', 'test']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: DataLoader(image_datasets[x],
                                      batch_size=flags['batch_size'],
                                      shuffle=True,
                                      num_workers=flags['num_workers'],
                                      pin_memory=True) for x in ['train', 'dev']}
    # Create non-shuffle Test dataloader separately
    dataloaders_dict['test'] = DataLoader(image_datasets['test'],
                                          batch_size=flags['batch_size'],
                                          shuffle=False,
                                          num_workers=flags['num_workers'],
                                          pin_memory=True)
    return dataloaders_dict
