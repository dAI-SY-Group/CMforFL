# Collaboration Management for Federated Learning
This repository contains the implementation of the Collaborator Matching and the corresponding federated training experiments.

The paper including all empirical results can be found on [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10555076)


## Please cite as:
```
@inproceedings{schlegel2024collaboration,
  title={Collaboration Management for Federated Learning},
  author={Schlegel, Marius and Scheliga, Daniel and Sattler, Kai-Uwe and Seeland, Marco and M{\"a}der, Patrick},
  booktitle={2024 IEEE 40th International Conference on Data Engineering Workshops (ICDEW)},
  pages={291--300},
  year={2024},
  organization={IEEE}
}
```

## Abstract:
Federated learning (FL) enables collaborative and privacy-preserving training of machine learning (ML) models on federated data. However, the barriers to using FL are still high. First, collaboration procedures are use-case-specific and require manual preparation, setup, and configuration of execution environments. Second, establishing collaborations and matching collaborators is time-consuming due to heterogeneous intents as well as data properties and distributions. Third, debugging the process and keeping track of the artifacts created and used during collaboration is challenging. Our goal is to reduce these barriers by requiring as little technical knowledge from collaborators as possible. We contribute mechanisms for flexible collaboration composition and creation, automated collaborator matching, and provenance-based collaboration and artifact management.

## Requirements:
You can create a [conda](https://www.anaconda.com/) virtual environment with the following packages:
```
conda create -n CMFL python=3.11.3 \
  pytorch=1.13.1 \
  cudatoolkit=11.8 \
  cudnn=8.8.0.121 \
  torchmetrics \
  torchvision \
  torchinfo \
  dill \
  pandas \
  munch \
  matplotlib \
  seaborn \
  pyyaml \
  prettytable
conda activate CMFL
pip install fedlab
```
or install it using the provided environment.yaml:
```
conda env create -f environment.yaml
```

## Usage:
We provide a demo notebook `FingerprintMatchingDemo.ipynb` to compute PACFL and FFP dataset fingerprints and visualize the similarity matrices used for collaborator matching.

Furthermore `federated_training.py` can be used to perform Clustered Federated Learning (CFL) with various configurations. 
We provide the configurations for our matched collaboration trainings in `configs/experiments/`.
These configurations are based on multiple base-configuration files. 
These can be found in `configs/bases/`.
To change specific parameters for the training process, adjust the corresponding base-configuration files.
An optional `--debug` flag can be set for debugging purposes (reduces the amount of communication rounds and epochs of training).
```
python federated_training.py configs/experiments/<config_file>.yaml (--debug)
```


## Credits:
We base our implementation on the following repositories:
+ [1] [GitHub](https://github.com/MMorafah/PACFL) for the implementation of [PACFL](https://ojs.aaai.org/index.php/AAAI/article/view/26197)
+ [2] [GitHub](https://github.com/dAI-SY-Group/FFPforCFL) for the implementation of [FFP](https://www.tandfonline.com/doi/full/10.1080/08839514.2024.2394756) and the federated training experiments.