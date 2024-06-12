# CTIF-Net: A CNN-Transformer Iterative Fusion Network for Salient Object Detection


> **Authors:** 
> Junbin Yuan,
> Aiqing Zhu,
> Qingzhen Xu,
> Kanoksak Wattanachote,
> Yongyi Gong

## Preface

- This repository provides code for "_**CTIF-Net: A CNN-Transformer Iterative Fusion Network for Salient Object Detection**_" [IEEE Transactions on Circuits and Systems for Video Technology, 2023](URL "[title](https://ieeexplore.ieee.org/abstract/document/10268450/)").


# Framework
![image](https://github.com/danielfaster/CTIF-Net/blob/main/figures/framework.png)


# Experiment
1. Visual comparison results
![image](https://github.com/danielfaster/CTIF-Net/blob/main/figures/visual_comparsion.png)

2. Quantitative comparison results
![image](https://github.com/danielfaster/CTIF-Net/blob/main//figures/quantitative_comparsion.png)


# Usage

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single GeForce RTX 3090Ti GPU of 24 GB Memory.

1. Requirements
    * pytorch 1.8.1
    * timm 1.0.3
2. Clone the repo
```
git clone https://github.com/danielfaster/CTIF-Net.git 
cd CTIF-Net
```
3. Train/Test
    * Train
        * Download the pre-trained checkpoint ViT-Large from [Link](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth) and move it to `./models/checkpoint/`
        * Download datasets: [DUTS-TR](http://saliencydetection.net/duts/)
        * Move it into `./data/TrainDataset/`, then 
        ```
        python train.py
        ```
    * Test
        * Download the pre-trained model from [Google Drive](https://drive.google.com/file/d/1SyhGknKiH7vLdN7DYAK4BgghetU7rEoy/view?usp=drive_link) or [Baidu Pan](https://pan.baidu.com/s/1C7C3_hAYv5biYVodRqOD_g?pwd=eyij), and put it in './models/'. This model is only trained on the training set of DUTS and tested on other datasets.
        * Download datasets: [DUTS-TE](http://saliencydetection.net/duts/), [DUT-OMRON](http://saliencydetection.net/dut-omron/), [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html), [PASCAL-S](http://www.cbi.gatech.edu/salobj/)
        * Move them into `./data/TestDatasets/`, then 
        ```
        python test.py
        ```
        * You can also download the pre-computed saliency maps from [Google Drive](https://drive.google.com/file/d/1245xp9yBcqysp5dJweS8o1oENOSQ_tov/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1cPC-xveHlKfQ9LgR4saamQ?pwd=q2sp).
 
## 3. Citation

Please cite our paper if you find the work useful: 

	@article{yuan2023ctif,
	  title={CTIF-Net: A CNN-Transformer Iterative Fusion Network for Salient Object Detection},
	  author={Yuan, Junbin and Zhu, Aiqing and Xu, Qingzhen and Wattanachote, Kanoksak and Gong, Yongyi},
	  journal={IEEE Transactions on Circuits and Systems for Video Technology},
	  year={2023},
	  publisher={IEEE}
	}