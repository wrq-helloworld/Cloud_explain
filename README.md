# Interpreting and Manipulating Uncertain Facial Semantics in GANs through Gaussian Cloud Distribution
## Requirements
* Python 3
* Matplotlib 3.2.2
* Scipy 1.5
* Numpy 1.18
## Data Setting
```
The data of latent variable from GANs latent space with shape (num, dim), where num is the number of latent variables, dim represents the dimension of the latent space.
-latent_var.npy
The label of latent_var.npy with shape (num, y), where num is the number of latent variables, y represents the number of semantics. 
-latent_concat_label_9.csv
```
## Build Gaussian Cloud Distribution
```
$ python Build_cloud_distribution.py
```

## Acknowledgement

We benifit a lot from [pytorch_GAN_zoo](https://github.com/facebookresearch/pytorch_GAN_zoo), thanks for their excellent work.
