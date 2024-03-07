# Learning Regularization Parameter-Maps with NNs

Here, we provide code for our paper

"Learning Regularization Parameter-Maps for Variational Image Reconstruction using Deep Neural Networks and Algorithm Unrolling"
https://arxiv.org/abs/2301.05888

by A. Kofler, F. Altekrüger, F. A. Ba, C. Kolbitsch, E. Papoutsellis, D. Schote, C. Sirotenko, F. F. Zimmermann, K. Papafitsoros.

The paper presents a Neural Networks (NNs)-based method to learn spatio-temporal pixel/voxel dependent and data-adaptive regularization parameter-maps to be used for variational image reconstruction. In the paper, we use total variation (TV)-minimization as the regularization method of choice.

## The Method
The overall approach works as follows. The NN consists of two sub-networks. In the first step, we estimate a regularization parameter-map ![equation](https://latex.codecogs.com/svg.image?\boldsymbol{\Lambda}_{\Theta}) from an input image ![equation](https://latex.codecogs.com/svg.image?\mathbf{x}_0) with  the first sub-network, i.e. ![equation](https://latex.codecogs.com/svg.image?\boldsymbol{\Lambda}_{\Theta}&space;=&space;\mathrm{NET}_{\Theta}(\mathbf{x}_0)), where ![equation](https://latex.codecogs.com/svg.image?\mathrm{NET}_{\Theta}) denotes a CNN, e.g. a U-Net. Then, we use the obtained regularization parameter-map ![equation](https://latex.codecogs.com/svg.image?\boldsymbol{\Lambda}_{\Theta}) to formulate the reconstruction problem

![equation](https://latex.codecogs.com/svg.image?\underset{\mathbf{x}}{\mathrm{min}}&space;\frac{1}{2}\|\|&space;\mathbf{A}\mathbf{x}&space;&space;&space;-\mathbf{z}\|\|_2^2&space;&plus;&space;&space;\|\|&space;\boldsymbol{\Lambda}_{\Theta}\nabla\mathbf{x}\|\|_1)

which we then (approximately) solve with an unrolled iterative scheme of finite length, e.g. the primal dual hybrid gradient (PDHG) method, which constitutes the second sub-network, see the following figure.

<img src="figures/pdhg_net.png" width="70%"/>

The weights of the network, i.e. the weights of the CNN that provides the regularization parameter-maps as well as other potential parameters as step sizes, can then be learned on a set of input-target image pairs. Thereby, the CNN learns to estimate appropriate regularization parameter-maps such that the TV-reconstruction is as close as possible to the corresponding target image. In contrast to other methods, we do not need access to target regularization parameter-maps.

Below, you can see exemplary results for a 2D dynamic MR image reconstruction problem. The regularization parameter-map ![equation](https://latex.codecogs.com/svg.image?\boldsymbol{\Lambda}_{\Theta}) is constructed as ![equation](https://latex.codecogs.com/svg.image?\boldsymbol{\Lambda}_{\Theta}:=(\boldsymbol{\Lambda}_{\Theta}^{xy},\boldsymbol{\Lambda}_{\Theta}^{xy},\boldsymbol{\Lambda}_{\Theta}^{t}) ), i.e. it consists of two spatial components ![equation](https://latex.codecogs.com/svg.image?\boldsymbol{\Lambda}_{\Theta}^{xy}) (which are constraint to be equal) for the derivatives in the spatial ![equation](https://latex.codecogs.com/svg.image?x)- and ![equation](https://latex.codecogs.com/svg.image?y)-direction and a temporal component ![equation](https://latex.codecogs.com/svg.image?\boldsymbol{\Lambda}_{\Theta}^{t}) for the derivative in the temporal ![equation](https://latex.codecogs.com/svg.image?t)-direction. The regularization parameter-maps denote the voxel-dependent strength of the gradient-sparsity in the spatial and temporal directions which is imposed on the image, respectively.

From left to right:

Zero-filled reconstruction - TV-reconstruction - spatial component ![equation](https://latex.codecogs.com/svg.image?\boldsymbol{\Lambda}_{\Theta}^{xy}) - temporal component of ![equation](https://latex.codecogs.com/svg.image?\boldsymbol{\Lambda}_{\Theta}^{t}) - ground-truth target image:

<img src="gifs/dyn_mri/xu_kz20.gif" width="19%"/> <img src="gifs/dyn_mri/xTV_kz20.gif" width="19%"/> <img src="gifs/dyn_mri/Lambda_cnn_xy_kz20.gif" width="19%"/> <img src="gifs/dyn_mri/Lambda_cnn_t_kz20.gif" width="19%"/>  <img src="gifs/dyn_mri/xf_kz20.gif" width="19%"/>

## Code and Applications

You will find code for the applications shown in the paper, i.e.
- 2D dynamic cardiac MRI
- 2D dynamic image denoising
- quantitative MRI (T1-mapping in the brain)
- 2D low-dose CT

We are currently cleaning and preparing the code for publication. Please forgive the time delay.

<b><u>START OF TODO</u></b>

<b><u>TODO: Update the versioning!!!</u></b>

(Version 1.1 (accepted version) - October 23st, 2023)

![summary_figure](summary_figure.png)

All test ... simulations can be found at <https://dx.doi.org/...>.
---

# Get started with runnning the model

<b><u>TODO: Update the dataset download!!!</u></b>

* Download the dataset: <https://dx.doi.org/...>

```bash
wget --no-check-certificate https://zenodo.org/api/records/.../files-archive

unzip files-archive

unzip raw_datasets.zip

mv raw_datasets/* database/raw_datasets/

rm files-archive

rm raw_datasets.zip

rm -rf raw_datasets
```

* Install the required libraries:

          pip install -r requirements.txt

<b><u>TODO: Update the dataset conversion and training!!!</u></b>

* **IMPORTANT:** Convert the download dataset into pickle files: run **create_dataset.ipynb** inside the **database** folder

* Explore the other notebooks! Try starting with **try_model.ipynb**

* For reproducing the paper's results, you can run **plot_results.ipynb**

* For training the model

        python main.py

---

## Repository

<b><u>TODO: Update the repository structure!!!</u></b>

The repository is divided in the following folders:

* **database:** Creation of hydrodynamic simulations (**D-Hydro simulations.ipynb**) [requires the license and installation of "D-HYDRO Suite 2023.01 1D2D"] and conversion of the NETCDF output files into PyTorch Geometric-friendly data (**create_dataset.ipynb**).
Also contains the output of the hydrodynamic simulations (**raw_datasets**: for downloading this dataset go to <https://dx.doi.org/10.5281/zenodo.7764418>). This is converted into Pickle files that are then stored and separated into training and testing datasets in **datasets**.

* **models:**  Deep learning models developed for surrogating the hydraulic one: contains MLP, CNN, and GNNs, as well as a base class with common inputs and functions.

* **results:** Contains trained models and respective configuration files, used for the paper's results.

* **training:** Contains loss functions, Trainer object, and testing functions.

* **utils:** Contains Python functions for loading, creating and scaling the dataset. There are also other miscellaneous functions and visualization functions.

This version of the repository is not very robust to changes in inputs, so be careful to adapt the utils.dataset functions when you want to apply the model to a new dataset!
The next version is much more flexible and also works with meshes, but it will be published later on, hehe.

<b></u>END OF TODO</u></b>


## Citation and Acknowledgement

The paper is accepted for publication in SIAM Journal of Imaging Sciences (SIIMS) and currently in press. Until then, please use the arXiv reference.

If you use the code for your work or if you found the code useful, please cite:

@article{kofler2023learning,
  title={Learning Regularization Parameter-Maps for Variational Image Reconstruction using Deep Neural Networks and Algorithm Unrolling},
  author={Kofler, Andreas and Altekr{\"u}ger, Fabian and Ba, Fatima Antarou and Kolbitsch, Christoph and Papoutsellis, Evangelos and Schote, David and Sirotenko, Clemens and Zimmermann, Felix Frederik and Papafitsoros, Kostas},
  journal={arXiv preprint arXiv:2301.05888},
  year={2023}
}

