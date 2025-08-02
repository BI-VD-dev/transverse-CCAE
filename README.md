[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
# tranverse-CCAE Project
This repository contains the core codebase for a PyTorch implementation of a conditional convolutional autoencoder for inferring the downstream transverse profile from an upstream measurement and magnetic optics.
The project supports modular experimentation, version control, and collaborative development via GitHub.

## Authors
 * Joseph Wolfenden - joseph.wolfenden@cockcroft.ac.uk

## Project Structure
```text
transverse-CCAE/
│
├── README.md ← You're here
├── requirements.txt ← Python dependencies
├── .gitignore ← Ignored files/folders including most outputs from the code
├── output/ ← Latest trained models
├── data_loader.py ← data loader class
├── ema.py ← EMA model class
├── generate_sample_images.py ← script to generate 10 sample images from model during/after training
├── model.py ← PyTorch model class for the CCAE
├── train.py ← Main code for training CCAE model
└── inference_demo.ipynb ← jupyter notebook for testing trained model
```
## Getting Started

1. **Clone your fork of the repository**
   ```bash
   git clone https://github.com/<your-username>/transverse_CCAE.git
   cd transverse_CCAE

2. **Setup virtual environment**

   Use python 3.10.18

3. **Install dependencies**
    ```bash
   pip install -r requirements.txt
   
4. **Data**

   This project is not currently tracking the large data files. These must be acquired from a shared folder from the authors.

## Trained Models
These are stored within the output folder. The EMA version is the model which has been used for validation and testing.

## Workflow
This project uses a fork-based workflow for development:

 * Fork the repo to your account
 * Create a new branch for each feature or experiment
 * Submit a Pull Request back to the BI-VD-dev master

## License
This project is licensed under the Creative Commons Attribution 4.0 International License.  
See the [LICENSE](LICENSE) file for details.
