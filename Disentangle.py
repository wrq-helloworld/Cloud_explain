import torch
import csv
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import wasserstein_distance as wd
import torch
from scipy.integrate import quad
from tqdm import tqdm

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
    return result

# Define the integration interval [a, c]
a = 0.0
c = 1.0
# The percentage of the dimension involved in manipulation
Ratio = 0.95
# The dimension of the latent space
dim = 131072
# The number of semantics
Gran_COUNT = 9
R_dim = [[[] for i in range(Gran_COUNT)] for i in range(Gran_COUNT)]

Ex_ = np.load('./Ex.npy')
En_ = np.load('./En.npy')
He_ = np.load('./He.npy')

print(Ex.shape,En.shape,He.shape)

for i in tqdm(range(Gran_COUNT)):
    for j in range(i, Gran_COUNT):
        if i == j:
            R_dim[i][i].append(0)
        else:
            distance = []
            distance_temp = 0
            for k in range(Ex.shape[0]):

                def gaussian_function(x):
                    return 1 - math.sqrt((2 * (En[k][i] - 3 * He[k][i] + 6 * He[k][i] * x) * (
                                En[k][j] - 3 * He[k][j] + 6 * He[k][j] * x) / (
                                                  pow((En[k][i] - 3 * He[k][i] + 6 * He[k][i] * x), 2) + pow(
                                              (En[k][j] - 3 * He[k][j] + 6 * He[k][j] * x), 2))) * math.pow(math.e, (
                        -math.pow((Ex[k][i] - Ex[k][j]), 2)) / (4 * (
                            pow((En[k][i] - 3 * He[k][i] + 6 * He[k][i] * x), 2) + pow(
                        (En[k][j] - 3 * He[k][j] + 6 * He[k][j] * x), 2)))))


                distance_temp, _ = quad(gaussian_function, a, c)

                distance.append(distance_temp)

            distance_temp = np.sort(distance)
            # print(distance_temp[0],distance_temp[7000])
            phi = distance_temp[int(dim * Ratio)]

            for k in range(Ex.shape[0]):
                if distance[k] <= phi:
                    R_dim[i][j].append(k)
                    R_dim[j][i].append(k)

            # print(len(R_dim[i][j]))

R_dim_result = [[[] for i in range(Gran_COUNT)] for i in range(Gran_COUNT)]
for i in range(Gran_COUNT):
    for j in range(i, Gran_COUNT):
        temp = []
        for k in range(Gran_COUNT):
            if k != i and k != j:
                temp1 = list(set(R_dim[i][k]) & set(R_dim[j][k]))
                if len(temp) == 0:
                    temp = temp1
                else:
                    temp = list(set(temp) & set(temp1))
                # print(i,j,k,len(D_reduce[i][k]),len(D_reduce[j][k]),len(temp1),len(temp))
        R_dim_result[i][j] = temp
        R_dim_result[j][i] = temp


R_dim_save = []
for i in range(Gran_COUNT):
    for j in range(Gran_COUNT):
        R_dim_save.append(R_dim_result[i][j])
        # print(np.array(R_dim_result[i][j]).shape, np.array(R_dim[i][j]).shape)
print(len(R_dim_save))

write_csv('./reduce_dim_' + str(int(Ratio * 100)) + '.csv',R_dim_save)






