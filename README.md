# NICE: Non-linear Independent Components Estimation

## Overview

This project implements the NICE flow model as described in the paper "NICE: Non-linear Independent Components Estimation." The model comprises four additive coupling layers, alternating between odd and even dimensions and fixing one set of dimensions at a time. The final layer includes a scaling layer that learns a scale parameter per dimension and scales by exp(s) with a small epsilon for stability. The latent space uses a logistic distribution.

## Dependencies

- Python (>=3.6)
- PyTorch (>=1.7.1)
- torchvision (>=0.8.2)
- matplotlib (>=3.3.4)
- numpy (>=1.19.2)

## Implementation

The code consists of the following components:

1. **Model Architecture:** The NICE flow model is implemented with 4 additive coupling layers and a scaling layer. Affine coupling can be used as an alternative.

2. **Dequantization:** Images from MNIST and Fashion-MNIST datasets are dequantized by adding uniform(0,1) random noise and rescaled to the [0, 1] range.

3. **Training:** The model is trained for 50 epochs on both MNIST and Fashion-MNIST datasets. The log-likelihood for each epoch is recorded for both training and testing.

4. **Results:** Plots of train and test log-likelihood for each epoch are generated for both datasets, considering both additive and affine coupling. Additionally, sampled images from the trained model are saved at the end of training.

## Instructions

1. Clone the repository:

   ```bash
   git clone <repository_url>
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:

   ```bash
   python train_nice.py
   ```

4. View the results:

   Plots and sampled images will be saved in the `results/` directory.

## Additional Information

- **Dequantization:** Dequantization is performed to convert discrete images into continuous ones. This is done by adding uniform(0,1) random noise and rescaling to the [0, 1] range.

- **Evaluation:** The evaluation of generative models follows the approach described in "A note on the evaluation of generative models sec. 3.1."

## Results

The results include plots of train and test log-likelihood for each epoch for both datasets with both additive and affine coupling. Additionally, sampled images from the trained model are provided.

## Acknowledgments

This implementation is based on the NICE flow model proposed in the paper "NICE: Non-linear Independent Components Estimation."

---
