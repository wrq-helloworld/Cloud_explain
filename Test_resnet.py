import os
import time
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
# import torchsummary
import csv
import numpy as np


def load_traindata(path, BATCH_SIZE, IMAGE_SIZE):
    '''load data from our dataset'''
    trainset = torchvision.datasets.ImageFolder(path, transform=transforms.Compose(
        [transforms.Resize((IMAGE_SIZE[1], IMAGE_SIZE[2])),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
         transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return trainloader

def write_csv(path, data):
    with open(path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(data)):
            writer.writerow(data[i])

def read_csv(path):
    result = []
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for r in reader:
            result.append(list(map(float, r)))
    return np.array(result)


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 10
result = []
test_path = "./generate//"
model_type = {"hair":5,"gender":2,"age":2}

# label gender information
subspace = "gender"
num_class = model_type[subspace]
test_loader = load_traindata(test_path, BATCH_SIZE, (3,128,128))
resnet18 = models.resnet18()
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, num_class)
resnet18.load_state_dict(torch.load("./resnet_model//resnet_CelebA_" + subspace + ".pkl"))
resnet18 = resnet18.to(device)
result_gender = []
for step, (b_x, b_y) in enumerate(test_loader):  # gives batch data, normalize x when iterate train_loader
    pre = resnet18(b_x.to(device))  # cnn output
    pred_y = torch.max(pre, 1)[1].cpu().data.numpy()
    result_gender = result_gender + pred_y.tolist()

# label hair information
subspace = "hair"
num_class = model_type[subspace]
test_loader = load_traindata(test_path, BATCH_SIZE, (3,128,128))
resnet18 = models.resnet18()
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, num_class)
resnet18.load_state_dict(torch.load("./resnet_model//resnet_CelebA_" + subspace + ".pkl"))
resnet18 = resnet18.to(device)
result_hair = []
for step, (b_x, b_y) in enumerate(test_loader):  # gives batch data, normalize x when iterate train_loader
    pre = resnet18(b_x.to(device))  # cnn output
    pred_y = torch.max(pre, 1)[1].cpu().data.numpy()
    result_hair = result_hair + pred_y.tolist()

# label age information
subspace = "age"
num_class = model_type[subspace]
test_loader = load_traindata(test_path, BATCH_SIZE, (3,128,128))
resnet18 = models.resnet18()
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, num_class)
resnet18.load_state_dict(torch.load("./resnet_model//resnet_CelebA_" + subspace + ".pkl"))
resnet18 = resnet18.to(device)
result_age = []
for step, (b_x, b_y) in enumerate(test_loader):  # gives batch data, normalize x when iterate train_loader
    pre = resnet18(b_x.to(device))  # cnn output
    pred_y = torch.max(pre, 1)[1].cpu().data.numpy()
    result_age = result_age + pred_y.tolist()

for i in range(len(result_gender))
    temp = []
    if result_gender[i] == 0:
        temp.append(0)
        temp.append(-1)
    else:
        temp.append(-1)
        temp.append(1)

    if result_hair[i] == 0:
        temp.append(2)
        temp.append(-1)
        temp.append(-1)
        temp.append(-1)
        temp.append(-1)
    elif result_hair[i] == 1:
        temp.append(-1)
        temp.append(3)
        temp.append(-1)
        temp.append(-1)
        temp.append(-1)
    elif result_hair[i] == 2:
        temp.append(-1)
        temp.append(-1)
        temp.append(4)
        temp.append(-1)
        temp.append(-1)
    elif result_hair[i] == 3:
        temp.append(-1)
        temp.append(-1)
        temp.append(-1)
        temp.append(5)
        temp.append(-1)
    elif result_hair[i] == 4:
        temp.append(-1)
        temp.append(-1)
        temp.append(-1)
        temp.append(-1)
        temp.append(6)

    if result_age[i] == 0:
        temp.append(7)
        temp.append(-1)
    else:
        temp.append(-1)
        temp.append(8)

    result.append(temp)

write_csv('./latent_concat_label_9.csv',result)