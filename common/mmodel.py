import torch
from torchvision import datasets, transforms
import pandas as pd
import os
from PIL import Image

from data_json.converter import make_dir

train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_model(model, path="pretrained"):
    make_dir(path)
    torch.save(model.state_dict(), os.path.join(path, "model_weights.pth"))
    torch.save(model, os.path.join(path, "model.pth"))

def load_model(path="pretrained"):
    model = torch.load(os.path.join(path, "model.pth"))
    model.load_state_dict(torch.load(os.path.join(path, "model_weights.pth")))
    return model

def create_dataloaders():
    data_transforms = {
        'train': train_transforms,
        'val': val_transforms,
    }

    data_dir = "data"
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                shuffle=True, num_workers=0)
                for x in ['train', 'val']}

    return dataloaders

def get_classes(path="data_json/names.csv", sep=","):
    classes = pd.read_csv(path, sep=sep).iloc[:,1]
    return classes.tolist()

def pred_images(model, images):
    model.eval()

    inputs = torch.empty([len(images), 3, 224, 224])
    for i, img in enumerate(images):
        img = Image.fromarray(img)
        inp = val_transforms(img)
        inputs[i] = inp
    inputs = inputs.to(get_device())
    outputs = model(inputs)
    _, predicts = torch.max(outputs, 1)
    return predicts.tolist()