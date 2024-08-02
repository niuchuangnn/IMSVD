# Variable Discretization for Self-Supervised Learning

This is a Pytorch implementation of the [paper](https://openreview.net/pdf?id=p7DIDSzT8x).


## Installation
Assuming [Anaconda](https://www.anaconda.com/) with python 3.8, a step-by-step example for installing this project is as follows:

```shell script
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
conda install matplotlib
```

## Pretrain on ImageNet

```shell script
sbatch job_vd.sh  # need to adjust according to your computing platform
```

## Linear Evaluation on ImageNet

```shell script
python linear_eval.py
```

## Visualization

Our pretrained model can be downloaded [here](https://drive.google.com/file/d/1NU_V3ZXuGWcRE14btg4K_kCAKpwi-o0t/view?usp=sharing)

```shell script
python visualize_matrix.py  # visualize the joint probability matrix
```

```shell script
python visualize_samples.py  # visualize samples assigned to specific feature units
```

### License
This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.