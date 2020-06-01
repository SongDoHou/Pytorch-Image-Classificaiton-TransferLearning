import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from PIL import Image
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import os
import argparse


def get_model(num_nodes, num_classes):
    model = models.resnext50_32x4d(True)
    for i, p in enumerate(model.parameters()):
        if i > 105:
            p.requires_grad = True
        else:
            p.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(0.55),
        nn.Linear(model.fc.in_features, num_nodes),
        nn.ReLU(),
        nn.Dropout(0.45),
        nn.Linear(num_nodes, num_classes)
    )
    return model

if __name__ == "__main__":
    T = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(0.35),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    T_test = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    cwd = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--train_dir', help="Train Folder Path")
    parser.add_argument('-b', '--batch_size', help="Batch size", type=int)
    parser.add_argument('-n', '--num_nodes', help="Number of nodes in hidden layer", type=int, default=512)
    parser.add_argument('-e', '--num_epochs', help="Number of epochs to training", type=int, default=30)
    parser.add_argument('-t', '--test_dir', help="Test Folder Path", default="N")
    args = parser.parse_args()
    train_data_path = os.path.join(cwd, args.train_dir)
    dataset = datasets.ImageFolder(train_data_path, T)
    model = get_model(args.num_nodes, len(dataset.classes)).cuda()
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if args.test_dir != "N":
        test_dataset = dataset.ImageFolder(os.path.join(cwd, args.test), T_test)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 0.0001, betas=(0.9, 0.999), weight_decay=0.003)
    best_accuracy = 0
    for i in range(0, args.num_epochs):
        for j, (img, label) in enumerate(dataloader):
            img = img.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            res = model(img)
            loss = loss_fn(res, label)
            loss.backward()
            optimizer.step()
        print("Epoch:{0}  Loss:{1}".format(i, loss))
        if args.test != "N":
            accuracy = 0
            with torch.no_grad():
                for j, (img, label) in enumerate(test_dataloader):
                    img = img.cuda()
                    sb = model(img)
                    model_answer = torch.max(sb, 1)[1].cpu()
                    if model_answer == label:
                        accuracy += 1
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    print("Epoch: {0}\tBest accuracy, saving model".format(i))
                    state = {"epoch": i, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                    torch.save(state, "./epoch{0}_resNext_50_v2".format(i))

                print("Correct: {0}\n Total:{1}\n Accuracy:{2}".format(accuracy,
                                                                       len(test_dataset),
                                                                       100. * (accuracy / len(test_dataset))))