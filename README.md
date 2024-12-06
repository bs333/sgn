# Robust Classification via Regression for Learning with Noisy Labels

This repository contains resources and implementations for the project **"Robust Classification via Regression for Learning with Noisy Labels"**, which explores a regression-based approach to tackle noisy labels in classification tasks. The project is built around benchmark datasets like CIFAR-10, CIFAR-100, and Fashion-MNIST.

---

## Table of Contents
1. [Overview](#overview)
2. [Project Components](#project-components)
3. [Dependencies](#dependencies)
4. [Usage Instructions](#usage-instructions)
5. [Results Summary](#results-summary)
6. [Applications](#applications)
7. [Citation](#citation)

---

## Overview

Deep neural networks are highly sensitive to noisy labels, leading to reduced performance. This project proposes a unified regression-based framework that combines **loss reweighting** and **label correction** techniques to improve classification robustness.

### Key Contributions:
- Novel use of **log-ratio transformations** for label correction.
- Integration of **Gaussian noise modeling** for robust regression.
- Superior performance compared to state-of-the-art baselines on noisy datasets.

---

## Project Components

### 1. Presentation
- **File**: `Robust Classification via Regression for Learning with Noisy Labels.pdf`
- **Description**: A detailed presentation covering methodology, experiments, and findings.

### 2. Python Scripts
- **`main_train.py`**:
  - Trains models on CIFAR-10 and CIFAR-100 datasets using the proposed approach.
  - Implements Gaussian noise modeling and log-ratio transformations.
- **`main_test.py`**:
  - Evaluates trained models on noisy test datasets.
  - Outputs performance metrics like accuracy and standard deviation.
- **`sgn-paper.py`**:
  - Implements the **Shifted Gaussian Noise (SGN)** framework with advanced noise modeling.

### 3. Results
- Experimental results comparing SGN and the implemented method on CIFAR-10, CIFAR-100, and Fashion-MNIST.

---

## Dependencies

Use the following `requirements.txt` file to install dependencies:

```bash
tensorflow==2.10.0
numpy
pandas
matplotlib
scipy

