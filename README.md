# Robust Classification via Regression for Learning with Noisy Labels

This repository contains the resources and code for the project **"Robust Classification via Regression for Learning with Noisy Labels"**. The project explores a regression-based framework for handling noisy labels in classification tasks, with applications on datasets like CIFAR-10, CIFAR-100, and Fashion-MNIST.

---

## Table of Contents
1. [Overview](#overview)
2. [Project Components](#project-components)
3. [Dependencies](#dependencies)
4. [Usage Instructions](#usage-instructions)
5. [Results Summary](#results-summary)
6. [Citation](#citation)

---

## Overview

Noisy labels in training data can lead to significant degradation in the performance of deep neural networks. This project proposes a regression-based framework that combines **loss reweighting** and **label correction** for robust classification. The methodology has been tested on benchmark datasets with synthetic and asymmetric noise.

Key Contributions:
- A novel **log-ratio transformation** for noisy label correction.
- Integration of **Gaussian noise modeling** in a regression framework.
- Superior performance on noisy datasets, outperforming state-of-the-art baselines.

---

## Project Components

### 1. Presentation
The presentation is available as a PDF file:
- **File**: `Robust Classification via Regression for Learning with Noisy Labels.pdf`
- **Description**: Discusses the methodology, results, and potential applications of the proposed framework.

### 2. Python Files
- **`main_train.py`**:
  - Trains the model on CIFAR-10 and CIFAR-100 datasets with noisy labels.
  - Includes log-ratio transformations and Gaussian noise modeling.

- **`main_test.py`**:
  - Tests the trained model on the CIFAR datasets.
  - Outputs performance metrics (accuracy, standard deviation) at various noise levels.

- **`sgn-paper.py`**:
  - Implements the **Shifted Gaussian Noise (SGN)** framework for CIFAR-100.
  - Demonstrates label smoothing, custom loss functions, and a WideResNet architecture.

---

## Dependencies

To set up the environment, use the `requirements.txt` file:

```bash
pip install -r requirements.txt
