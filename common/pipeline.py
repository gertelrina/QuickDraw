from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
from itertools import cycle

from common import mobilenetv3, pipeline, mmodel

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, dataset_sizes, num_epochs=25):
    since = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(next(model.parameters()).is_cuda)
    waiting = ["-", "/", "\\"]
    waiting = cycle(waiting)
    mmodel.save_model(model, "pretrained/best_model")
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            total_batch_num = (dataset_sizes[phase] + 63) // 64
            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                #print(f"\r{waiting[i//3]}")
                print(f"\r{next(waiting)} {i/total_batch_num*100:.2f}%", end="")
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('\r{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                mmodel.save_model(model, "pretrained/best_model")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model = mmodel.load_model("pretrained/best_model")
    return model

def train_mnetv3(num_epochs=25, pretrained=False):
    dataloaders = mmodel.create_dataloaders()
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    class_names = dataloaders['train'].dataset.classes

    device = mmodel.get_device()

    if pretrained:
        model_ft = mmodel.load_model()
    else:
        model_ft = mobilenetv3.mobilenetv3_large(num_classes=len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=7)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = pipeline.train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, device, dataset_sizes, num_epochs=num_epochs)
    return model_ft