import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import os
import config
from torchvision.utils import save_image
import time
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

def write_csv(path, data, type='a'):
    with open(path, type, newline='') as csvfile:
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

def to_img(x):
    # out = 0.5 * (x + 1)
    out = x.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 1, 28, 28)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out

def GC_2(data):
    Ex = np.mean(data,axis=0)
    En_pre = data - Ex
    c_2 = np.mean(En_pre * En_pre, axis=0)
    c_4 = np.mean(En_pre * En_pre * En_pre * En_pre, axis=0)
    En = np.power((9*c_2*c_2-c_4)/6,0.25)
    He = np.power(c_2-np.power((9 * c_2 * c_2 - c_4)/6,0.5), 0.5)
    return Ex, En, He

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':

    path_p_space_w = './latent_var.npy'           # The data of latent variable
    lable_path = './latent_concat_label_9.csv'    # The lable of latent variable
    Gran_COUNT = 9                                # The number of semantics
    # lable_path = './latent_concat_label_10_shoes_D.csv'

    p_space = np.load(path_p_space_w)
    lable = read_csv(lable_path)

    p_space_choose = [[] for i in range(Gran_COUNT)]

    for l in range(lable.shape[0]):
        # print(lable[l][0])
        # lable_choose[int(lable[l][0])].append(lable[l][0])
        for j in range(Gran_COUNT):

            if lable[l][j] >= 0:
                p_space_choose[int(lable[l][j])].append(p_space[l])

        for i in range(Gran_COUNT):
            print(len(p_space_choose[i]))

    # print()
    '''Gaussian '''
    isfirst = True
    for i in range(Gran_COUNT):
        gm_0 = GaussianMixture(n_components=1, random_state=0).fit(p_space_choose[i])
        miu_0 = np.array(gm_0.means_).reshape((-1))
        covariances_0 = np.array(gm_0.covariances_)
        covariances_0 = covariances_0.reshape((covariances_0.shape[1],covariances_0.shape[2]))
        sigma_0 = np.diag(covariances_0)
        if isfirst:
            p_space_mu = miu_0[:, np.newaxis]
            p_space_sigma = sigma_0[:, np.newaxis]
            p_space_cov = covariances_0[:, :, np.newaxis]
            isfirst = False
        else:
            p_space_mu = np.concatenate((p_space_mu, miu_0[:, np.newaxis]), axis=1)
            p_space_sigma = np.concatenate((p_space_sigma, sigma_0[:, np.newaxis]), axis=1)
            p_space_cov = np.concatenate((p_space_cov, covariances_0[:, :, np.newaxis]), axis=2)

        print(p_space_mu.shape,p_space_sigma.shape,p_space_cov.shape)

    En = np.zeros((p_space_mu.shape[0], p_space_mu.shape[1]))
    He = np.zeros((p_space_mu.shape[0], p_space_mu.shape[1]))

    for i in tqdm(range(mu.shape[0])):
        # gender
        beta = abs(p_space_mu[i][0] - p_space_mu[i][1]) / (3 * (p_space_sigma[i][0] + p_space_sigma[i][1]))   # alpha
        En[i][0] = (1 + beta) * p_space_sigma[i][0] / 2
        He[i][0] = (1 - beta) * p_space_sigma[i][0] / 6
        En[i][1] = (1 + beta) * p_space_sigma[i][1] / 2
        He[i][1] = (1 - beta) * p_space_sigma[i][1] / 6

        # age
        beta = abs(p_space_mu[i][7] - p_space_mu[i][8]) / (3 * (p_space_sigma[i][7] + p_space_sigma[i][8]))
        En[i][7] = (1 + beta) * p_space_sigma[i][7] / 2
        He[i][7] = (1 - beta) * p_space_sigma[i][7] / 6
        En[i][8] = (1 + beta) * p_space_sigma[i][8] / 2
        He[i][8] = (1 - beta) * p_space_sigma[i][8] / 6

        # hair
        temp = np.zeros((mu.shape[1]) - 4)
        for j in range(2, 7):
            order = 0
            for k in range(2, 7):
                if p_space_mu[i][j] > p_space_mu[i][k]:
                    order += 1
            temp[order] = j
        for j in range(temp.shape[0]):
            if j == 0:
                beta = (p_space_mu[i][int(temp[j + 1])] - p_space_mu[i][int(temp[j])]) / (
                        3 * (p_space_sigma[i][int(temp[j + 1])] + p_space_sigma[i][int(temp[j])]))
            elif j == temp.shape[0] - 1:
                beta = (p_space_mu[i][int(temp[j])] - p_space_mu[i][int(temp[j - 1])]) / (
                        3 * (p_space_sigma[i][int(temp[j])] + p_space_sigma[i][int(temp[j - 1])]))
            else:
                beta_1 = (p_space_mu[i][int(temp[j + 1])] - p_space_mu[i][int(temp[j])]) / (
                        3 * (p_space_sigma[i][int(temp[j + 1])] + p_space_sigma[i][int(temp[j])]))
                beta_2 = (p_space_mu[i][int(temp[j])] - p_space_mu[i][int(temp[j - 1])]) / (
                        3 * (p_space_sigma[i][int(temp[j])] + p_space_sigma[i][int(temp[j - 1])]))
                beta = min(beta_1, beta_2)
            En[i][int(temp[j])] = (1 + beta) * sigma[i][int(temp[j])] / 2
            He[i][int(temp[j])] = (1 - beta) * sigma[i][int(temp[j])] / 6

    np.save('./Ex.npy', p_space_mu)
    np.save('./En.npy', En)
    np.save('./He.npy', He)

