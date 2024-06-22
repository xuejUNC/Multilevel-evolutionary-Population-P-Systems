# Multilevel-evolutionary-Population-P-Systems
### Platform

The code runs on a Linux with a 10GB GPU, cuDNN 8.0.2 or above, and has at least 6 CPU cores (12 threads) when trained.The proposed HN P system was also implemented on the PyTorch framework on an NVIDIA Tesla V100 GPU with 32 GB. The ensemble learning is conducted by three HN P systems with different parameters on three numbers of NVIDIA Tesla V100 GPU with 32 GB simultaneously.

### Installation

The installation procedure of the core computation of MEPP system, i.e., nnunet can be found on git clone https://github.com/MIC-DKFZ/nnUNet.git

### Dataset

The dataset are in .nii format:1)The NIH pancreas segmentation dataset Pancreas-CT: https://wiki.cancerimagingarchive.net/display/Public/ ;2)the BraTS 2020 training dataset, which contains 369 cases with gliomas. There are four MRI modalities for each case: T1-weighted (T1), postcontrast T1-weighted (T1Gd), T2-weighted (T2) and T2 fluid-attenuated inversion recovery (FLAIR):Menze, B.H., et al., The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). IEEE Trans Med Imaging, 2015. 34(10): p. 1993-2024.

### Core computation of MEPP system

Functions of nnUNet are adopted as core computations for segmenting organs/tumors.

generic_UNet.py is the framework of the overall network, which call the base "initialization.py" and "neural_network.py"


### control variable T

Inspired by the simulated annealing algorithm, where the state and energy of matter are controlled, reconstructed and optimized by temperature, we design control variable T to guide the optimization of MEPP system in subpopulation membranes automatically.

In the file "control variable T", the training is implemented in network_training. In network_training, the control variable T is conducted in nnUNetTrainerV2-DRF.py to guide the optimization of MEPP system. And nnUNetTrainerV2-final.py call the other py files to complete the training.


### Ensemble learning of MEPP system

In the file "ensemble learning of MEPP system", The predict.py call the predict_simple.py and segmentation_export.py to segment organs/tumors. Five MEPP system with different parameters are conducted on five numbers of NVIDIA Tesla V100 GPU with 32 GB simultaneously. The final model is the best results ensembled from the validation datasets. Which output the final segmentation.


