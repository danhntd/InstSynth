# InstSynth: Instance-wise Prompt-guided Style Masked Conditional Data Synthesis for Scene Understanding

This repository is the official implementation of the paper entitled: **InstSynth: Instance-wise Prompt-guided Style Masked Conditional Data Synthesis for Scene Understanding**. <br>
**Authors**: Thanh-Danh Nguyen, Bich-Nga Pham, Trong-Tai Dam Vu, Vinh-Tiep Nguyen† , Thanh Duc Ngo, and Tam V. Nguyen.

[[Paper]]() [[Code]](https://github.com/danhntd/InstSynth)

---
## Updates
[2024/7] We have released the visualization, and initial instructions for InstSynth⚡!

## 1. Environment Setup
Download and install Anaconda with the recommended version from [Anaconda Homepage](https://www.anaconda.com/download): [Anaconda3-2019.03-Linux-x86_64.sh](https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh) 
 
```
git clone https://github.com/danhntd/InstSynth.git
cd InstSynth
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
```

After completing the installation, please create and initiate the workspace with the specific versions below. The experiments were conducted on a Linux server with a single `GeForce RTX 2080Ti GPU`, CUDA 11.1, Torch 1.9.

```
conda create --name InstSynth python=3
conda activate InstSynth
conda install pytorch==1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia -y
```

This source code is based on [Detectron2](https://github.com/facebookresearch/detectron2). Please refer to INSTALL.md for the pre-built or building Detectron2 from source.

After setting up the dependencies, use the command `python setup.py build develop` in this root to finish.

In case you face some environmental conflicts, these installations may help:
```
pip install mxnet-mkl==1.6.0 numpy==1.23.1
```

## 2. Data Preparation
In this work, we utilize Cityscapes Dataset for training and testing our proposed method.

### Download the datasets

Please visit [this link](https://www.cityscapes-dataset.com/) for the dataset description and downloading.

## 3. Training Pipeline
Our proposed InstSynth framework:
<img align="center" src="/visualization/framework.png">


## 4. Visualization

<p align="center">
  <img width="800" src="/visualization/visualization.png">
</p>

## Citation
Please use the following bibtex to cite this repository:
```
@inproceedings{nguyen2024instsynth,
  title={InstSynth: Instance-wise Prompt-guided Style Masked Conditional Data Synthesis for Scene Understanding},
  author={Nguyen, Thanh-Danh and Pham, Bich-Nga and Dam Vu, Trong-Tai and Nguyen, Vinh-Tiep and Ngo, Thanh Duc and Tam, Nguyen V.},
  booktitle={2023 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```

## Acknowledgements

[OneFormer](https://github.com/SHI-Labs/OneFormer.git) [FastInst](https://github.com/junjiehe96/FastInst.git) [Detectron2](https://github.com/facebookresearch/detectron2.git) 