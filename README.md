# An Information Theoretic Approach to Unlearning

![GitHub last commit (branch)](https://img.shields.io/github/last-commit/jwf40/Information-Theoretic-Unlearning/main) ![GitHub Repo stars](https://img.shields.io/github/stars/jwf40/Information-Theoretic-Unlearning) ![GitHub repo size](https://img.shields.io/github/repo-size/jwf40/Information-Theoretic-Unlearning)


![SSD_heading](/assets/jit_ITU.png)


This is the code for the paper **[An Information Theoretic Approach to Machine Unlearning](https://browse.arxiv.org/abs/2402.01401)**.

_Note: VGG16 in the code is actually a VGG11. Working on updating these refs._

## Usage

Experiments can be run via the shell script (0 is the GPU number, feel free to change on a multi-gpu setup; 1 is the seed - set to any int).

```
./cifar100_fullclass_exps_vgg.sh 0 1
```
You might encounter issues with executing this file due to different line endings with Windows and Unix. Use dos2unix "filename" to fix.

## Setup

You will need to either download the weights we used for our models from **[here](https://drive.google.com/drive/folders/1B2-A5tOnjbNGOpNM_LmG0zj2xO7o8rsF)**, or train VGG16 and Vision Transformers from scratch using pretrain_model.py for this and then copy the paths of the models into the respecive .sh files. **Important**: Due to neural net's propensity to learn smooth functions first, please make sure to train the models for a sufficient amount of time. Premature stopping significantly impacts unlearning performance.

```
# fill in _ with your desired parameters as described in pretrain_model.py
python pretrain_model.py -net _ -dataset _ -classes _ -gpu _
```

We used https://hub.docker.com/layers/tensorflow/tensorflow/latest-gpu-py3-jupyter/images/sha256-901b827b19d14aa0dd79ebbd45f410ee9dbfa209f6a4db71041b5b8ae144fea5 as our base image and installed relevant packages on top.

```
datetime
wandb
sklearn
torch
copy
tqdm
transformers
matplotlib
scipy
```

You will need a wandb.ai account to use the implemented logging. Feel free to replace with any other logger of your choice.

## Modifying JiT unlearning

JiT functions are in Lipschitz.py, and is referred to throughout as lipschitz_forgetting. To change sigma and eta, set them in the respective forget_..._main.py file per unlearning task.

<!-- ## Citing this work

```
@misc{foster2023fast,
      title={Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening}, 
      author={Jack Foster and Stefan Schoepf and Alexandra Brintrup},
      year={2023},
      eprint={2308.07707},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
``` -->

## Authors

For our newest research, feel free to follow our socials:

Jack Foster: [LinkedIn](https://www.linkedin.com/in/jackfoster-ml/), [Twitter](https://twitter.com/JackFosterML) , [Scholar](https://scholar.google.com/citations?user=7m8cBAoAAAAJ&hl=en)

Kyle Fogarty: [LinkedIn](https://www.linkedin.com/in/kylefogarty), [Twitter](https://twitter.com/ktfogarty), [Scholar](https://scholar.google.com/citations?hl=en&user=yEwwq4EAAAAJ)

Stefan Schoepf: [LinkedIn](https://www.linkedin.com/in/schoepfstefan/), [Twitter](https://twitter.com/S__Schoepf), [Scholar](https://scholar.google.com/citations?hl=en&user=GTvLmf0AAAAJ)

Cengiz Öztireli: [LinkedIn](https://www.linkedin.com/in/cengizoztireli/), [Scholar](https://scholar.google.com/citations?hl=en&user=dXt1WOUAAAAJ)

Alexandra Brintrup: [LinkedIn](https://www.linkedin.com/in/alexandra-brintrup-1684171/), [Scholar](https://scholar.google.com/citations?hl=en&user=8HJL8cAAAAAJ)

Supply Chain AI Lab: [LinkedIn](https://www.linkedin.com/company/supply-chain-ai-lab/)  
