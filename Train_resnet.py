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


def load_traindata(path, BATCH_SIZE, IMAGE_SIZE):
    '''load data from our dataset'''
    trainset = torchvision.datasets.ImageFolder(path,transform=transforms.Compose([transforms.Resize((IMAGE_SIZE[1], IMAGE_SIZE[2])),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                                                    transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True, num_workers=0)
    return trainloader

def write_csv(path, data):
    with open(path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(data)):
            writer.writerow(data[i])

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCH = 100               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 10
LR = 0.001              # learning rate

subspace = "hair"                                                      # Semantic subspace, options hair, gender, age
train_path = "./data//CelebA_" + subspace + "//"                       # In folder CelebA_ + subspace, images with different semantic information are placed in different folders.
validation_path = "./data//CelebA_" + subspace + "_Validation//"       #

model_type = {"hair":5,"gender":2,"age":2}
num_class = model_type[subspace]
train_loader = load_traindata(train_path, BATCH_SIZE, (3,128,128))
validation_loader = load_traindata(validation_path, BATCH_SIZE, (3,128,128))

resnet18 = models.resnet18(pretrained=True)
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, num_class)
resnet18 = resnet18.to(device)

optimizer = torch.optim.Adam(resnet18.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()
accuracy_list = []
acc_last = 0
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        pre = resnet18(b_x.to(device))               # cnn output
        loss = loss_func(pre, b_y.to(device))   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        print('Epoch: ', epoch, 'Step: ', step, '| train loss: %.4f' % loss.cpu().data.numpy())
        # print(output.cpu().data.numpy()[:,:10])

    with torch.no_grad():
        accuracy = 0
        for step_test, (t_x, t_y) in enumerate(validation_loader):  # gives batch data, normalize x when iterate train_loader
            test_pre = resnet18(t_x.to(device))
            pred_y = torch.max(test_pre, 1)[1].cpu().data.numpy()
            accuracy += float((pred_y == t_y.cpu().data.numpy()).astype(int).sum()) / float(t_y.size(0))
        accuracy = accuracy / (step_test + 1)
        accuracy_list.append([accuracy])
        print('Epoch: ', epoch, '| test accuracy: %.2f' % accuracy)
        write_csv('./accuracy_list.csv',accuracy_list)
        if accuracy > acc_last:
            torch.save(resnet18.state_dict(), "./resnet_model//resnet_CelebA_" + subspace + ".pkl")
            acc_last = accuracy


