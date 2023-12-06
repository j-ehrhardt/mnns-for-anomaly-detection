![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
<!-- ![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?style=for-the-badge&logo=ResearchGate&logoColor=white) -->


# Modular Neural Networks for Anomaly Detection in Cyber-Physical Systems

This is the repository for our publication *Using Modular Neural Networks for Anomaly Detection in Cyber-Physical Systems*.

## Overview

We use Modular Neural Networks to model the inner dependencies of Cyber-Physical System (CPS) subsystems.
Thereby, we can achieve a more robust detection of anomalies in CPS and a better allocation of their root-causes.
For further information, we recommend you to read our [publication(#citation)].

## Table of Contents

- [Requirements](#requirements)
- [Replication](#replication)
- [Citation](#citation)
- [License](#license)

## Requirements

We recommend Anaconda to install all requirements for our repository.
The requirements are saved in the `venv.yml` file. 

For a quick installation run: `conda env create -f venv.yml`

## Replication

As empirical validation dataset, we used the robot-anomaly dataset of [Grabaek et al. 2023](https://doi.org/10.1109/access.2023.3289068). 
You can access and download the dataset [here](https://zenodo.org/record/5849300).
Once you downloaded the dataset, save it in the `./data` directory.

For replicating the results from our paper, run the `main.py` script from the `./code` directory.
The script will run the reproducible hyperparameter search, as well as the subsequent replication studies, and evaluation studies.

By running the `./exp/exp_setup/evaluation.ipynb`, you can calculate the metrics from the paper.

You can run your own studies by uncommenting the suitable codeblock in the `main.py` script.


## Citation

When using this work, please cite: 

```
@inproceedings{Ehrhardt2024,
    title={Using Modular Neural Networks for Anomaly Detection in Cyber-Physical Systems},
    author={Ehrhardt, Jonas and Overlöper, Phillip and Vranjes, Daniel and Steude, Henrik and Diedrich, Alexander and Niggemann, Oliver},
    year={2024},
}
```

## LICENSE

Licensed under MIT license


<img src="https://www.hsu-hh.de/imb/wp-content/uploads/sites/677/2021/01/IMB_1080.png" width="100" height="100" alt="IMB">
