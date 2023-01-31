# Hybrid-Neural-like-P-Systems-
### Platform

The code runs on a Linux with a 10GB GPU, cuDNN 8.0.2 or above, and has at least 6 CPU cores (12 threads) when trained.The proposed HN P system was also implemented on the PyTorch framework on an NVIDIA Tesla V100 GPU with 32 GB. The ensemble learning is conducted by three HN P systems with different parameters on three numbers of NVIDIA Tesla V100 GPU with 32 GB simultaneously.

### Installation

The installation procedure of the core computation of HN P system, i.e., nnunet can be found on git clone https://github.com/MIC-DKFZ/nnUNet.git

### Dataset

The dataset are in .nii format and the .json file of 5006 BMs of 288 patients from Shandong Cancer Hospital (named the SCH dataset) from August 2018 to April 2021.

### Core computation of HN P system

Functions of nnUNet are adopted as core computations for segmenting BMs.

generic_UNet.py is the framework of the overall network, which call the base "initialization.py" and "neural_network.py"


### Dynamic regulatory factors

Inspired by the simulated annealing algorithm [45], where the state and energy of matter are controlled, reconstructed and optimized by temperature, we design two dynamic regulatory factors to guide the optimization of HN P systems automatically.

In the file "dynamic regulatory factors", the training is implemented in network_training. In network_training, the dynamic regulatory factors are conducted in nnUNetTrainerV2-DRF.py to guide the optimization of HN P systems. And nnUNetTrainerV2-final.py call the other py files to complete the training.


### Ensemble learning of HN P system

In the file "ensemble learning of HN P system", The predict.py call the predict_simple.py and segmentation_export.py to segment BMs. TThree HN P systems with different parameters are conducted on three numbers of NVIDIA Tesla V100 GPU with 32 GB simultaneously. The final model is the best results ensembled from the validation datasets. Which output the final segmentation.


