from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from torchvision import transforms

import pytorch_lightning as pl
from torchvision import models


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class CustomCNN(nn.Module):
    def __init__(self, output_dim):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3)),  # 8 x 222 x 222
            nn.MaxPool2d(4, stride=3),  # 8 x 73 x 73
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3)),  # 16 x 71 x 71
            nn.MaxPool2d(3, stride=2),  # 16 x 35 x 35
            nn.Dropout2d(),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),  # 32 x 33 x 33
            nn.MaxPool2d(3, stride=2),  # 32 x 16 x 16
            nn.Dropout2d(),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(32*16*16, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, output_dim)
        self.drop = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class SmallLinear(nn.Module):
    def __init__(self, trf_dim, obj_dim, output_dim):
        super(SmallLinear, self).__init__()
        self.fc = nn.Linear(trf_dim + obj_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


def init_pretrained_model(model_name, output_dim, feature_extract, use_pretrained=True, extract_features=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        if extract_features:
            new_classifier = nn.Sequential(*list(model_ft.classifier.children())[:-2])
            model_ft.classifier = new_classifier
        else:
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, output_dim)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        if extract_features:
            model_ft.classifier = model_ft.classifier[:-3]
        else:
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, output_dim)
        input_size = 224

    elif model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        if extract_features:
            modules = list(model_ft.children())[:-1]
            model_ft = nn.Sequential(*modules)
        else:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, output_dim)
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        if extract_features:
            model_ft.classifier = nn.Identity()
        else:
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, output_dim)
        input_size = 224

    elif model_name == "custom":
        """ Custom CNN
        """
        model_ft = CustomCNN(output_dim)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


class LightningTransfer(pl.LightningModule):

    def __init__(self, model, flags):
        super().__init__()
        self.model = model
        self.flags = flags
        self.class_weights = torch.Tensor(flags['class_weight'])
        if self.flags['multilabel']:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)  # combines sigmoid + bceloss
        else:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)  # combines logsoftmax + nllloss

    def forward(self, x):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x).flatten(1)
        outputs = torch.sigmoid(outputs)
        predictions = torch.where(outputs > self.flags['threshold'], 1, 0)
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = y.type_as(y_hat) if self.flags['multilabel'] else y.long().squeeze_()
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = y.type_as(y_hat) if self.flags['multilabel'] else y.long().squeeze_()
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y = y.type_as(y_hat) if self.flags['multilabel'] else y.long().squeeze_()
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.flags['learning_rate'])
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]


def init_cls_model(flags, checkpoint_path=None):
    model_ft, input_size = init_pretrained_model(flags['model_name'], flags['output_dim'], flags['feature_extract'],
                                                 use_pretrained=flags['use_pretrained'])
    if not checkpoint_path:
        return LightningTransfer(model_ft, flags), input_size
    else:
        return LightningTransfer.load_from_checkpoint(checkpoint_path, model=model_ft, flags=flags), input_size


def init_small_model(flags, checkpoint_path = None):
    model_ft = SmallLinear(flags['trf_dim'], flags['obj_dim'], flags['output_dim'])
    if not checkpoint_path:
        return LightningTransfer(model_ft, flags)
    else:
        return LightningTransfer.load_from_checkpoint(checkpoint_path, model=model_ft, flags=flags)


if __name__ == '__main__':

    FLAGS = {
        'model_name': 'alexnet',
        'output_dim': 2,
        'batch_size': 512,  # 512 / N gpu
        'num_workers': 4,  # 4 per gpu
        'learning_rate': 0.02,  # 0.02 * N gpu
        'max_epochs': 10,  # arbitrary
        'threshold': 0.5,
        'feature_extract': True,
        'use_pretrained': True,
        'multilabel': True
    }
    trfs = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open('1.jpg')
    ex = trfs(image)
    ex = torch.unsqueeze(ex,0)
    # print(ex)
    m = CustomCNN(6)
    m(ex)
