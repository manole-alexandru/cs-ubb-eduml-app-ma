import argparse
import io
import warnings
from pathlib import Path

import torchvision
import json
import os
import shutil
import numpy as np
import math
import pickle
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import zipfile
from torch.optim.lr_scheduler import OneCycleLR
import copy

import numpy as np
import pandas as pd
from minio import Minio
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from PIL import Image
from scipy.io import loadmat

import mlflow.sklearn

from cs_ubb_eduml_app_ma.config import Settings
from cs_ubb_eduml_app_ma.mlflow.wrappers import sklearn_model, torch_model

settings = Settings.from_env()
warnings.filterwarnings("ignore")
np.random.seed(40)

LOSS_WEIGHTS = [0.9, 0.1]


ROOT_DIR = Path(__file__).parent.parent.parent


def load_data() -> pd.DataFrame:
    if settings.minio.enabled:
        minio = Minio(
            settings.minio.uri,
            access_key=settings.minio.user,
            secret_key=settings.minio.password,
            secure=False
        )
        response = minio.get_object(settings.minio.bucket, settings.minio.path)
        try:
            return pd.read_csv(io.StringIO(response.data.decode()))
        finally:
            response.close()
            response.release_conn()
    else:
        local_path = ROOT_DIR / "data" / "dataset.csv"
        with open(local_path, "r"):
            result = pd.read_csv(local_path)
        return result


def eval_metrics(actual, pred) -> dict:
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return {
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
    }

class ClassicMobileNetV3(nn.Module):
    def __init__(self, num_classes=100, pretrained=True, dropout_p=0.5, MODEL=None, MODEL_NAME='mobilenet_v3'):
        super(ClassicMobileNetV3, self).__init__()

        # Load the pre-trained MobileNetV3 model from torchvision
        self.mobilenet = MODEL(pretrained=pretrained)
        # print(self.mobilenet)

        if 'mobilenet_v3' in MODEL_NAME:

          # Modify the first convolutional layer to accept (28, 28) input size
          # self.mobilenet.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)

          # Extract the number of features from the original classifier
          in_features = self.mobilenet.classifier[-1].in_features

          # Replace the classifier with a new sequence including Dropout
          self.mobilenet.classifier[-1] = nn.Sequential(
              # nn.Dropout(p=dropout_p),  # Dropout layer with probability p
              nn.Linear(in_features, num_classes)  # Final fully connected layer
          )

        elif 'resnet' in MODEL_NAME:
          in_features = self.mobilenet.fc.in_features
          self.mobilenet.fc = nn.Linear(in_features, num_classes)
        elif 'swin' in MODEL_NAME:
          self.mobilenet.head = nn.Linear(self.mobilenet.head.in_features, num_classes)
        elif 'vit_b_16' in MODEL_NAME:
          self.mobilenet.heads.head = nn.Linear(self.mobilenet.heads.head.in_features, num_classes)
        elif 'vgg' in MODEL_NAME:
          # ? Redundant path
          in_features = self.mobilenet.classifier[-1].in_features
          self.mobilenet.classifier[-1] = nn.Linear(in_features, num_classes)
        elif 'densenet' in MODEL_NAME:
          in_features = self.mobilenet.classifier.in_features
          self.mobilenet.classifier = nn.Linear(in_features, num_classes)
        elif 'efficientnet' in MODEL_NAME:
          in_features = self.mobilenet.classifier[-1].in_features
          self.mobilenet.classifier[-1] = nn.Linear(in_features, num_classes)
        elif 'maxvit' in MODEL_NAME:
          in_features = self.mobilenet.classifier[-1].in_features
          self.mobilenet.classifier[-1] = nn.Linear(in_features, num_classes)
        elif 'convnext' in MODEL_NAME:
          in_features = self.mobilenet.classifier[-1].in_features
          self.mobilenet.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
          assert False


    def forward(self, x):
        return self.mobilenet(x)
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ParallelMultiTaskMobileNetV3(nn.Module):
    def __init__(self, num_superclasses=20, num_subclasses=100, pretrained=True, dropout_p=0.5, MODEL=None, MODEL_NAME='mobilenet_v3'):
        super(ParallelMultiTaskMobileNetV3, self).__init__()

        # Load the pre-trained MobileNetV3 model from torchvision
        self.mobilenet = MODEL(pretrained=pretrained)
        # Define Dropout layer
        self.dropout = nn.Dropout(p=dropout_p)

        if 'mobilenet_v3' in MODEL_NAME:
          # Modify the first convolutional layer to accept (28, 28) input size
          # self.mobilenet.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)

          # Extract the number of features from the original classifier
          in_features = self.mobilenet.classifier[-1].in_features

          # Replace the original classifier with an identity layer
          self.mobilenet.classifier[-1] = Identity()

        elif 'resnet' in MODEL_NAME:
          in_features = self.mobilenet.fc.in_features
          self.mobilenet.fc = Identity()

        elif 'swin' in MODEL_NAME:
          in_features = self.mobilenet.head.in_features
          self.mobilenet.head = Identity()

        elif 'vit_b_16' in MODEL_NAME:
          in_features = self.mobilenet.heads.head.in_features
          self.mobilenet.heads.head = Identity()

        elif 'vgg' in MODEL_NAME:
          # ? Redundant path
          in_features = self.mobilenet.classifier[-1].in_features
          self.mobilenet.classifier[-1] = Identity()

        elif 'densenet' in MODEL_NAME:
          in_features = self.mobilenet.classifier.in_features
          self.mobilenet.classifier = Identity()

        elif 'efficientnet' in MODEL_NAME:
          in_features = self.mobilenet.classifier[-1].in_features
          self.mobilenet.classifier[-1] = Identity()

        elif 'maxvit' in MODEL_NAME:
          in_features = self.mobilenet.classifier[-1].in_features
          self.mobilenet.classifier[-1] = Identity()

        elif 'convnext' in MODEL_NAME:
          in_features = self.mobilenet.classifier[-1].in_features
          self.mobilenet.classifier[-1] = Identity()

        self.superclass_classifier = nn.Linear(in_features, num_superclasses)
        self.subclass_classifier = nn.Linear(in_features, num_subclasses)

    def forward(self, x):
        # Pass input through the feature extractor
        features = self.mobilenet(x)
        features = torch.flatten(features, 1)  # Flatten the output for the classifier

        # Apply Dropout to the features
        # features = self.dropout(features)

        # Compute superclass and subclass predictions
        superclass_output = self.superclass_classifier(features)
        subclass_output = self.subclass_classifier(features)

        return subclass_output, superclass_output
    
class CascadedMultiTaskMobileNetV3(nn.Module):
    def __init__(self, num_superclasses=20, num_subclasses=100, pretrained=True, dropout_p=0.5, MODEL=None, MODEL_NAME='mobilenet_v3'):
        super(CascadedMultiTaskMobileNetV3, self).__init__()

        # Load the pre-trained MobileNetV3 model from torchvision
        self.mobilenet = MODEL(pretrained=pretrained)

        # Define Dropout layer
        self.dropout = nn.Dropout(p=dropout_p)

        # Modify the first convolutional layer to accept (28, 28) input size
        # self.mobilenet.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        if 'mobilenet_v3' in MODEL_NAME:
        # Adjust the classifier input size if needed (e.g., due to smaller input size)
          in_features = self.mobilenet.classifier[-1].in_features

          # Replace the original classifier with an identity layer
          self.mobilenet.classifier[-1] = Identity()

        elif 'resnet' in MODEL_NAME:
          in_features = self.mobilenet.fc.in_features
          self.mobilenet.fc = Identity()

        elif 'swin' in MODEL_NAME:
          in_features = self.mobilenet.head.in_features
          self.mobilenet.head = Identity()

        elif 'vit_b_16' in MODEL_NAME:
          in_features = self.mobilenet.heads.head.in_features
          self.mobilenet.heads.head = Identity()

        elif 'vgg' in MODEL_NAME:
          # ? Redundant path
          in_features = self.mobilenet.classifier[-1].in_features
          self.mobilenet.classifier[-1] = Identity()

        elif 'densenet' in MODEL_NAME:
          in_features = self.mobilenet.classifier.in_features
          self.mobilenet.classifier = Identity()

        elif 'efficientnet' in MODEL_NAME:
          in_features = self.mobilenet.classifier[-1].in_features
          self.mobilenet.classifier[-1] = Identity()

        elif 'maxvit' in MODEL_NAME:
          in_features = self.mobilenet.classifier[-1].in_features
          self.mobilenet.classifier[-1] = Identity()

        elif 'convnext' in MODEL_NAME:
          in_features = self.mobilenet.classifier[-1].in_features
          self.mobilenet.classifier[-1] = Identity()

        # Superclass classifier
        self.superclass_classifier = nn.Linear(in_features, num_superclasses)

        # Subclass classifier with additional input for superclass prediction
        self.subclass_classifier = nn.Linear(in_features + num_superclasses, num_subclasses)

    def forward(self, x):
        # Pass input through the feature extractor (all layers except the classifier)
        features = self.mobilenet(x)
        features = torch.flatten(features, 1)

        # Apply Dropout to the features
        # features = self.dropout(features)

        # Compute superclass prediction
        superclass_output = self.superclass_classifier(features)

        # Concatenate the superclass prediction with the features
        combined_input = torch.cat((features, superclass_output), dim=1)

        # Apply Dropout to the combined input before subclass classification
        # combined_input = self.dropout(combined_input)

        # Compute subclass prediction using combined input
        subclass_output = self.subclass_classifier(combined_input)

        return subclass_output, superclass_output

class StandfordCarsDataset(Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        """
        Args:
            data (any format): The data (e.g., list of file paths, array of images).
            labels (any format): The labels associated with the data.
            transform (callable, optional): Optional transform to apply to a sample.
        """
        self.root = root
        self.split = split
        if self.split == 'train' or self.split == 'val':
          self.path = 'cars_train'
        else:
          self.path = 'cars_train'

        self.image_paths = self.root / self.path / self.path 
        self.label_path = self.root / 'cars_annos.mat'
        annotation_path = self.root / 'cardatasettrain.csv' # if self.split == 'train' else '/content/dataset/cardatasettest.csv'
        self.annotation_df = pd.read_csv(annotation_path)
        self.data = []
        self.labels = []
        self.classes = []
        self.alphabetical_to_original_label = {}
        self.load_data()
        self.transform = transform
        self.target_transform = target_transform

    def load_data(self):
      data = loadmat(self.label_path)
      self.classes = data['class_names'][0]
      self.hierarchic_classes = list(set([cls[0].split(' ')[0] for cls in self.classes]))
      # no_samples = 8144 if self.split =='train' else 8041 # len(data['annotations'][0])
      no_samples = len(self.annotation_df)
      # for ids, sample in enumerate(data['annotations'][0]):
      for ids, sample in self.annotation_df.iterrows():
        # if (self.split == 'train' and ids < no_samples * 0.7) or (self.split == 'val' and (ids >= no_samples * 0.7 and ids < no_samples * 0.9)) or (self.split == 'test' and ids >= no_samples * 0.9):
        image_name =  sample['image']  # sample[0][0].split('/')[-1]
        image_no = int(image_name.split('.')[0])
        # if (self.split == 'train' and image_no < no_samples) or (self.split == 'val' and image_no > 8144 and image_no < 8144 + no_samples *0.8) or (self.split == 'test' and image_no > 8144 and image_no >= 8144 + no_samples *0.8):
        #  if self.split in ['val', 'test']:
        if (self.split == 'train' and ids <= no_samples * 0.7) or (self.split == 'val' and ids > no_samples * 0.7 and ids < no_samples * 0.8) or (self.split == 'test' and ids >= no_samples * 0.8):
            image_name = str(image_no).zfill(5) + '.jpg'
            self.data.append(self.image_paths / image_name)
            class_id = sample['Class'] - 1 # sample[5][0][0] - 1
            if class_id not in self.alphabetical_to_original_label:
              superclass = self.classes[class_id][0].split(' ')[0]
              self.alphabetical_to_original_label[class_id] = {'index': class_id, 'class': self.classes[class_id], 'super_index': self.hierarchic_classes.index(superclass), 'superclass':superclass}
            self.labels.append(class_id)

    def get_alphabetical_to_original_label(self):
       return self.alphabetical_to_original_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the sample data and label
        sample = Image.open(self.data[idx]).convert("RGB") # torchvision.io.read_image(self.data[idx])
        label = self.labels[idx]

        # Apply any transformation if provided
        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
          label = self.target_transform(label)

        return sample, label
    
def get_hierarchic_labels(target, alphabetical_to_original_label):
    superclass_label = alphabetical_to_original_label[target]['super_index']
    target = alphabetical_to_original_label[target]['index']
    return superclass_label, target

import torch
import copy
from torch.optim.lr_scheduler import OneCycleLR

def train_classic_mobilenet(model, dataloaders, criterion, optimizer, num_epochs=25, device='cuda', max_lr=1e-3, phases=['train', 'val']):
    model = model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'test_loss':[], 'test_acc':[]}

    # Calculate total steps for 1CycleLR (total number of batches in training phase)
    total_steps = len(dataloaders['train']) * num_epochs

    # Initialize the 1CycleLR scheduler
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(dataloaders['train']), epochs=num_epochs)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()  # Update the learning rate according to the 1Cycle policy

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the model if validation loss has decreased
            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                # best_model_wts = copy.deepcopy(model.state_dict())
                # model_save_path = BASE_DIR + f'classic_model_epoch_{epoch+1}.pt'
                # torch.save(model.state_dict(), model_save_path)
                print(f'New best model saved with validation loss: {best_val_loss:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model, history

def train_parallel_mobilenet(model, dataloaders, criterion, optimizer, num_epochs=25, device='cuda', max_lr=1e-3, phases=['train', 'val']):
    model = model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'train_acc_superclass': [], 'val_acc_superclass': [], 'test_loss':[], 'test_acc':[], 'test_acc_superclass':[]}

    # Calculate total steps for 1CycleLR (total number of batches in training phase)
    total_steps = len(dataloaders['train']) * num_epochs

    # Initialize the 1CycleLR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(dataloaders['train']), epochs=num_epochs)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_corrects_superclass = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                superclass_labels, subclass_labels = labels
                superclass_labels = superclass_labels.to(device)
                subclass_labels = subclass_labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    subclass_outputs, superclass_outputs = model(inputs)
                    superclass_loss = criterion(superclass_outputs, superclass_labels) * LOSS_WEIGHTS[1]
                    subclass_loss = criterion(subclass_outputs, subclass_labels) * LOSS_WEIGHTS[0]
                    loss = superclass_loss + subclass_loss

                    _, superclass_preds = torch.max(superclass_outputs, 1)
                    _, subclass_preds = torch.max(subclass_outputs, 1)
                    running_corrects += torch.sum(subclass_preds == subclass_labels.data).item()
                    running_corrects_superclass += torch.sum(superclass_preds == superclass_labels.data).item()

                    if phase == 'train':
                        superclass_loss.backward(retain_graph=True)
                        subclass_loss.backward()
                        optimizer.step()
                        scheduler.step()  # Update the learning rate according to the 1Cycle policy

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            epoch_acc_superclass = running_corrects_superclass / len(dataloaders[phase].dataset)

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            history[f'{phase}_acc_superclass'].append(epoch_acc_superclass)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Superclass Acc: {epoch_acc_superclass:.4f}')

            # Save the model if validation loss has decreased
            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                # model_save_path = BASE_DIR + f'parallel_model_epoch_{epoch+1}.pt'
                # torch.save(model.state_dict(), model_save_path)
                print(f'New best model saved with validation loss: {best_val_loss:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model, history

def train_cascaded_mobilenet(model, dataloaders, criterion, optimizer, num_epochs=25, device='cuda', max_lr=1e-3, phases=['train', 'val']):
    model = model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'train_acc_superclass': [], 'val_acc_superclass': [], 'test_loss':[], 'test_acc':[], 'test_acc_superclass':[]}

    # Calculate total steps for 1CycleLR (total number of batches in training phase)
    total_steps = len(dataloaders['train']) * num_epochs

    # Initialize the 1CycleLR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(dataloaders['train']), epochs=num_epochs)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_corrects_superclass = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                superclass_labels, subclass_labels = labels
                superclass_labels = superclass_labels.to(device)
                subclass_labels = subclass_labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    subclass_outputs, superclass_outputs = model(inputs)
                    superclass_loss = criterion(superclass_outputs, superclass_labels) * LOSS_WEIGHTS[1]
                    subclass_loss = criterion(subclass_outputs, subclass_labels) * LOSS_WEIGHTS[0]
                    loss = superclass_loss + subclass_loss

                    _, superclass_preds = torch.max(superclass_outputs, 1)
                    _, subclass_preds = torch.max(subclass_outputs, 1)
                    running_corrects += torch.sum(subclass_preds == subclass_labels.data).item()
                    running_corrects_superclass += torch.sum(superclass_preds == superclass_labels.data).item()

                    if phase == 'train':
                        superclass_loss.backward(retain_graph=True)
                        subclass_loss.backward()
                        optimizer.step()
                        scheduler.step()  # Update the learning rate according to the 1Cycle policy

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            epoch_acc_superclass = running_corrects_superclass / len(dataloaders[phase].dataset)

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            history[f'{phase}_acc_superclass'].append(epoch_acc_superclass)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Superclass Acc: {epoch_acc_superclass:.4f}')

            # Save the model if validation loss has decreased
            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                # model_save_path = BASE_DIR + f'cascaded_model_epoch_{epoch+1}.pt'
                # torch.save(model.state_dict(), model_save_path)
                print(f'New best model saved with validation loss: {best_val_loss:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model, history

@torch_model(settings.mlflow.enabled, settings.mlflow.tracking_uri, settings.mlflow.experiment_name)
def fit_standford_cars(epochs: int, lr: float, max_lr: float, model_name: str):
    EPOCHS = epochs
    DEVICE = 'cuda' # 'cpu'
    EXP_NO = 24
    LR = lr
    MAX_LR = max_lr
    MODEL_NAME = model_name
    AUGMENTATION = False

    criterion = nn.CrossEntropyLoss()

    if MODEL_NAME == 'mobilenet_v3_small':
        MODEL = torchvision.models.mobilenet_v3_small
    elif MODEL_NAME == 'mobilenet_v3_large':
        MODEL = torchvision.models.mobilenet_v3_large
    elif MODEL_NAME == 'resnet18':
        MODEL = torchvision.models.resnet18
    elif MODEL_NAME == 'resnet50':
        MODEL = torchvision.models.resnet50
    elif MODEL_NAME == 'swin_b':
        MODEL = torchvision.models.swin_s
    elif MODEL_NAME == 'vgg16':
        MODEL = torchvision.models.vgg16
    elif MODEL_NAME == 'vgg19':
        MODEL = torchvision.models.vgg19
    elif MODEL_NAME == 'densenet121':
        MODEL = torchvision.models.densenet121
    elif MODEL_NAME == 'vit_b_16':
        MODEL = torchvision.models.vit_b_16
    elif MODEL_NAME == 'efficientnet_b0':
        MODEL = torchvision.models.efficientnet_b0
    elif MODEL_NAME == 'maxvit_t':
        MODEL = torchvision.models.maxvit_t
    elif MODEL_NAME == 'convnext_base':
        MODEL = torchvision.models.convnext_base

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # torchvision.transforms.RandomRotation(15),
        torchvision.transforms.ToTensor(),
    #  torchvision.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),  # CIFAR-100 normalization values
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if not AUGMENTATION:
        train_transforms = test_transforms

    zip_path = ROOT_DIR / "data" / "standford_cars.zip" 
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(ROOT_DIR / "data")

    NUM_CLASSES = 196
    NUM_SUPERCLASSES = 49
    NUM_SUBCLASSES = 196

    DATASET_PATH = ROOT_DIR / "data"

    train_dataset = StandfordCarsDataset(root=DATASET_PATH, split='train', transform=train_transforms)
    alphabetical_to_original_label = train_dataset.alphabetical_to_original_label
    val_dataset = StandfordCarsDataset(root=DATASET_PATH, split='val', transform=test_transforms)
    test_dataset = StandfordCarsDataset(root=DATASET_PATH, split='test', transform=test_transforms)
    train_loader_classic  = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader_clasic = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader_clasic = DataLoader(test_dataset, batch_size=64, shuffle=False)

    train_dataset_hierarchic = StandfordCarsDataset(root=DATASET_PATH, split='train', transform=train_transforms, target_transform=lambda target: get_hierarchic_labels(target, alphabetical_to_original_label))
    val_dataset_hierarchic = StandfordCarsDataset(root=DATASET_PATH, split='val', transform=test_transforms, target_transform=lambda target: get_hierarchic_labels(target, alphabetical_to_original_label))
    test_dataset_hierarchic = StandfordCarsDataset(root=DATASET_PATH, split='test', transform=test_transforms, target_transform=lambda target: get_hierarchic_labels(target, alphabetical_to_original_label))

    train_loader_parallel = DataLoader(train_dataset_hierarchic, batch_size=32, shuffle=True, num_workers=4)
    val_loader_parallel = DataLoader(val_dataset_hierarchic, batch_size=32, shuffle=False, num_workers=4)
    test_loader_parallel = DataLoader(test_dataset_hierarchic, batch_size=32, shuffle=False, num_workers=4)

    # Initialize the model
    classic_model = ClassicMobileNetV3(num_classes=NUM_CLASSES, pretrained=True, MODEL=MODEL, MODEL_NAME=MODEL_NAME)

    # Define the loss function and optimizer
    optimizer = optim.Adam(classic_model.parameters(), lr=LR)

    # Call the training function for ClassicMobileNetV3
    trained_classic_model, classic_history = train_classic_mobilenet(
        model=classic_model,
        dataloaders={'train': train_loader_classic, 'val': val_loader_clasic, 'test':test_loader_clasic},
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=EPOCHS,
        device=DEVICE  # or 'cpu' if you don't have a GPU
    )

    parallel_model = ParallelMultiTaskMobileNetV3(num_superclasses=NUM_SUPERCLASSES, num_subclasses=NUM_SUBCLASSES, pretrained=True, MODEL=MODEL, MODEL_NAME=MODEL_NAME)
    optimizer = optim.Adam(parallel_model.parameters(), lr=LR)

    # Call the training function for ParallelMultiTaskMobileNetV3
    trained_parallel_model, parallel_history = train_parallel_mobilenet(
        model=parallel_model,
        dataloaders={'train': train_loader_parallel, 'val': val_loader_parallel, 'test':test_loader_parallel},
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=EPOCHS,
        device=DEVICE  # or 'cpu' if you don't have a GPU
    )

    cascaded_model = CascadedMultiTaskMobileNetV3(num_superclasses=NUM_SUPERCLASSES, num_subclasses=NUM_SUBCLASSES, pretrained=True, MODEL=MODEL, MODEL_NAME=MODEL_NAME)
    optimizer = optim.Adam(cascaded_model.parameters(), lr=LR)

    trained_cascaded_model, cascaded_history = train_cascaded_mobilenet(
        model=cascaded_model,
        dataloaders={'train': train_loader_parallel, 'val': val_loader_parallel, 'test':test_loader_parallel},
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=EPOCHS,
        device=DEVICE  # or 'cpu' if you don't have a GPU
    )

    return classic_history, parallel_history, cascaded_history

@sklearn_model(settings.mlflow.enabled, settings.mlflow.tracking_uri, settings.mlflow.experiment_name)
def fit_predict_wine_quality(a: float, l1: float):
    wine_quality_df = load_data()
    train, test = train_test_split(wine_quality_df)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    lr = ElasticNet(alpha=a, l1_ratio=l1, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    return (
        test_x,
        predicted_qualities,
        lr,
        eval_metrics(test_y, predicted_qualities),
    )


# Split the data into training and test sets. (0.75, 0.25) split.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs")
    parser.add_argument("--lr")
    parser.add_argument("--max_lr")
    parser.add_argument("--model_name")
    args = parser.parse_args()
    epochs = int(args.epochs)
    lr = float(args.lr)
    max_lr = float(args.max_lr)
    model_name = str(args.model_name)
    print(f"parsed args {epochs}, {lr}, {max_lr}, {model_name}")

    # fit_predict_wine_quality(alpha, l1_ratio)
    classic_history, parallel_history, cascaded_history = fit_standford_cars(epochs, lr, max_lr, model_name)
    print(classic_history, parallel_history, cascaded_history)
