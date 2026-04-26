import os
import numpy as np
import torch
import csv
from tqdm import tqdm


def write_csv(path, data):
    with open(path, "a", newline='') as csvfile:
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

def read_csv_list(path):
    result = []
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for r in reader:
            result.append(list(map(float, r)))
    return result

def semantic_change_certain(data, label, Ex, En, He, Gran_COUNT):

    result_all = []
    for i in tqdm(range(data.shape[0])):
        data_in = data[i]
        result = []
        for c in range(Gran_COUNT):
            data_change = np.zeros_like(data_in)
            ori = -1
            # gender
            if c == 0:
                ori = 1
            else:
                ori = 0
            # age
            if c == 7:
                ori = 8
            else:
                ori = 7
            # hair
            for ccc in range(5):
                if label[i][ccc + 2] >= 0:
                    ori = ccc + 2
                    break
                if ori == c:
                    if c == 2:
                        ori = 3
                    elif c == 3:
                        ori = 2
                    elif c == 4:
                        ori = 2
                    elif c == 5:
                        ori = 6
                    elif c == 6:
                        ori = 5
            for j in range(data.shape[1]):
                data_change[j] = (data_in[j] - Ex[j][ori]) / En[j][ori] * En[j][c] + \
                                             Ex[j][c]
                result.append(data_change)
        if i == 0:
            result_all = np.array(result)
        else:
            result_all = np.concatenate((result_all,result),axis=0)
    print(result_all.shape)
    # np.save('./cloud_changed.npy', result_all)
    np.save('./semantic_change_certain.npy', result_all)

def semantic_change_uncertain(data, label, Ex, En, He, Gran_COUNT):

    result_all = []
    for i in tqdm(range(data.shape[0])):
        data_in = data[i]
        result = []
        for c in range(Gran_COUNT):
            data_change = np.zeros_like(data_in)
            ori = -1
            # gender
            if c == 0:
                ori = 1
            else:
                ori = 0
            # age
            if c == 7:
                ori = 8
            else:
                ori = 7
            # hair
            for ccc in range(5):
                if label[i][ccc + 2] >= 0:
                    ori = ccc + 2
                    break
                if ori == c:
                    if c == 2:
                        ori = 3
                    elif c == 3:
                        ori = 2
                    elif c == 4:
                        ori = 2
                    elif c == 5:
                        ori = 6
                    elif c == 6:
                        ori = 5
            for j in range(data.shape[1]):
                Eno = random.gauss(En[j][ori], He[j][ori])
                Ent = random.gauss(En[j][c], He[j][c])
                data_change[j] = (data_in[j] - Ex[j][ori]) / Eno * Ent + Ex[j][c]

                result.append(data_change)
        if i == 0:
            result_all = np.array(result)
        else:
            result_all = np.concatenate((result_all,result),axis=0)
    print(result_all.shape)
    # np.save('./cloud_changed.npy', result_all)
    np.save('./semantic_change_uncertain.npy', result_all)

def semantic_change_certain_dis(data,label,Ratio, Ex, En, He, Gran_COUNT):
    reduce_dim = read_csv('./reduce_dim_' + str(int(Ratio * 100)) + '.csv')
    reduce = [[] for i in range(Gran_COUNT)]
    for i in range(len(reduce_dim)):
        reduce[int(i / Gran_COUNT)].append(reduce_dim[i])

    result_all = []
    for i in tqdm(range(data.shape[0])):
        data_in = data[i]
        result = []
        for c in range(Gran_COUNT):
            data_change = np.zeros_like(data_in)
            ori = -1
            # gender
            if c == 0:
                ori = 1
            else:
                ori = 0
            # age
            if c == 7:
                ori = 8
            else:
                ori = 7
            # hair
            for ccc in range(5):
                if label[i][ccc + 2] >= 0:
                    ori = ccc + 2
                    break
                if ori == c:
                    if c == 2:
                        ori = 3
                    elif c == 3:
                        ori = 2
                    elif c == 4:
                        ori = 2
                    elif c == 5:
                        ori = 6
                    elif c == 6:
                        ori = 5

                for j in range(data.shape[1]):
                    if j in reduce[ori][c]:
                        data_change[j] = (data_in[j] - Ex[j][ori]) / En[j][ori] * En[j][c] + \
                                                     Ex[j][c]
                    else:
                        data_change[j] = data_in[j]
            result.append(data_change)
        if i == 0:
            result_all = np.array(result)
        else:
            result_all = np.concatenate((result_all, result), axis=0)

    print(result_all.shape)
    np.save('./cloud_changed_certain_' + str(int(Ratio * 100)) + '.npy', result_all)


def semantic_change_uncertain_dis(data, label, Ratio, Ex, En, He, Gran_COUNT):
    reduce_dim = read_csv('./reduce_dim_' + str(int(Ratio * 100)) + '.csv')
    reduce = [[] for i in range(Gran_COUNT)]
    for i in range(len(reduce_dim)):
        reduce[int(i / Gran_COUNT)].append(reduce_dim[i])

    result_all = []
    for i in tqdm(range(data.shape[0])):
        data_in = data[i]
        result = []
        for c in range(Gran_COUNT):
            data_change = np.zeros_like(data_in)
            ori = -1
            # gender
            if c == 0:
                ori = 1
            else:
                ori = 0
            # age
            if c == 7:
                ori = 8
            else:
                ori = 7
            # hair
            for ccc in range(5):
                if label[i][ccc + 2] >= 0:
                    ori = ccc + 2
                    break
                if ori == c:
                    if c == 2:
                        ori = 3
                    elif c == 3:
                        ori = 2
                    elif c == 4:
                        ori = 2
                    elif c == 5:
                        ori = 6
                    elif c == 6:
                        ori = 5

                for j in range(data.shape[1]):
                    if j in reduce[ori][c]:
                        Eno = random.gauss(En[j][ori], He[j][ori])
                        Ent = random.gauss(En[j][c], He[j][c])
                        data_change[j] = (data_in[j] - Ex[j][ori]) / Eno * Ent + Ex[j][c]
                    else:
                        data_change[j] = data_in[j]
            result.append(data_change)
        if i == 0:
            result_all = np.array(result)
        else:
            result_all = np.concatenate((result_all, result), axis=0)

    print(result_all.shape)
    np.save('./cloud_changed_uncertain_' + str(int(Ratio * 100)) + '.npy', result_all)


data = np.load('./latent_var.npy')                 # The data of latent variable
label = read_csv('./latent_concat_label_9.csv')    # The lable of latent variable
Gran_COUNT = 9                                     # The number of semantics
# Parameters of the Gaussian cloud distribution
Ex = np.load('./Ex.npy')
En = np.load('./En.npy')
He = np.load('./He.npy')

# Uncertain Semantic Manipulation USM
semantic_change_certain(data, label, Ex, En, He, Gran_COUNT)
# Certain Semantic Manipulation CSM
semantic_change_uncertain(data, label, Ex, En, He, Gran_COUNT)

# The percentage of the dimension involved in manipulation
Ratio = 0.95
# Disentangled Uncertain Semantic Manipulation DUSM
semantic_change_all_certain_dis(data,label,Ratio, Ex, En, He, Gran_COUNT)
# Disentangled Certain Semantic Manipulation DCSM
semantic_change_all_uncertain_dis(data, label, Ratio, Ex, En, He, Gran_COUNT)
