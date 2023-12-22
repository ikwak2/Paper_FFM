# Enhancing Robustness of Voice Spoofing Detection Using Frequency Feature Masking and Comparative Augmentation Analysis

This repository contains the source for a manuscript "Enhancing Robustness of Voice Spoofing Detection Using Frequency Feature Masking and Comparative Augmentation Analysis."

A PDF will be available after publication.

Also see [our workshop paper](https://ikwak2.github.io/publications/ddam004-kwak.pdf). Our team ranked 3rd place at the ICASSP 2022 Grand Challenge on Audio Deepfake Detection, Track 1 (Low quality fake audio detection) using Frefuency Feature Masking. [Challenge Link](http://addchallenge.cn/#/) 


## Data Preparation
For training or evaluating on the Audio Spoof dataset you need to download the dataset (you can do it from the official repository) and then run the following script being located in the root folder of the project:

You will need to download the [ADD 2022 dataset](http://addchallenge.cn/download) and the [ASVspoof 2019 dataset](https://datashare.ed.ac.uk/handle/10283/3336) for training or evaluation.

```
python run_FFM_BCResMax.py
```
---

Here's a description of the files:

- FFM_Aug.ipynb: ipynb notebook file that demonstrate the use of FFM.
- etc

---

The manuscript is licensed under the
[Creative Commons Attribution 3.0 Unported License](http://creativecommons.org/licenses/by/3.0/).

[![CC BY](http://i.creativecommons.org/l/by/3.0/88x31.png)](http://creativecommons.org/licenses/by/3.0/)

The software is licensed under the [MIT license](License.md).
