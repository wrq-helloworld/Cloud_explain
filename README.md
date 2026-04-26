# Interpreting and Manipulating Uncertain Facial Semantics in GANs through Gaussian Cloud Distribution
## Requirements
* Python 3
* Matplotlib 3.2.2
* Scipy 1.5
* Numpy 1.18
## Data Setting
The CelebA datasets
```
-data
    -img1.jpg
    -img2.jpg
    -...
```
The data of latent variable from GANs latent space with shape (num, dim), num is the number of latent variables, dim represents the dimension of the latent space.
```
-latent_var.npy
```
The label of latent_var.npy with shape (num, y), num is the number of latent variables, y represents the number of semantics.
```
-latent_concat_label_9.csv
```
## Manipulation of the process
### 1. Train the GANs by CelebA datasets
We choose the GANs from [pytorch_GAN_zoo](https://github.com/facebookresearch/pytorch_GAN_zoo) in our works, and generate images (saved in folder "./generate") and their corresponding latent variables "latent_var.npy" by GANs.
### 2. Label the latent variables
In this work, the Resnet model is selected as the model for labeling the latent variables. Since the relationship between the latent variable and the generated image is one-to-one mapping, the latent variable can be labeled by analyzing the semantic information of the image. Use the CelebA datasets to train the Resnet model by "train_resnet.py".
```
$ python Train_resnet.py
```
Use the trained Resnet to label "latent_var.npy"  and obtain "latent_concat_label_9.csv" .
```
$ python Test_resnet.py
```
### 3. Calculate the parameters of semantic distribution
Construct the Gaussian cloud distributions based on heuristic Gaussian cloud transformation.
```
$ python Build_cloud_distribution.py
```
### 4. Disentangle semantics
The degree of disentanglement can be adjusted by setting the "Ratio", which is equivalent to $\epsilon$ in the paper. "Ratio" represents the proportion of the dimensions involved in the manipulation process.
```
$ python Disentangle.py
```
### 5. Manipulation semantics
All four types of semantic manipulation results in latent space, namely USM, DUSM, CSM, and DCSM, can be obtained.
```
$ python Semantic_change_cloud.py
```
### 6. Generate results
Input the obtained latent variables from the previous step into the GANs to obtain the manipulation results in the image space.
## Acknowledgement
We benifit a lot from [pytorch_GAN_zoo](https://github.com/facebookresearch/pytorch_GAN_zoo), thanks for their excellent work.
