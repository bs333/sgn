# Robust Classification via Regression for Learning with Noisy Labels

This repository contains resources and implementations for the project and paper **"Robust Classification via Regression for Learning with Noisy Labels"** by *Erik Englesson* and *Hossein Azizpour*, which explores a regression-based approach to tackle noisy labels in classification tasks. The project is built around benchmark datasets like CIFAR-10, CIFAR-100, and Fashion-MNIST.

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
```

Install them via:

```bash
pip install -r requirements.txt
```

## Usage Instructions

### Training

Run the training script to train the model:

```bash
python main_train.py
```

### Testing

Evaluate the model using:

```bash
python main_test.py
```

### Additional Notes

- Modify dataset paths in the scripts to point to CIFAR or Fashion-MNIST datasets.
- Checkpoints are saved in the `checkpoints` folder during training.

---

## Results Summary

### CIFAR-10 and CIFAR-100 Results

| Dataset       | Noise Type      | SGN Accuracy (%) | Our Implementation (%) |
|---------------|-----------------|------------------|-------------------------|
| **CIFAR-10**  | No Noise (0%)   | 94.12 ± 0.22     | 92.10 ± 0.25           |
|               | Symmetric (20%) | 93.02 ± 0.17     | 91.45 ± 0.20           |
|               | Symmetric (40%) | 91.29 ± 0.25     | 89.12 ± 0.30           |
|               | Symmetric (60%) | 86.03 ± 1.19     | 84.50 ± 1.10           |
| **CIFAR-100** | No Noise (0%)   | 73.88 ± 0.34     | 72.10 ± 0.40           |
|               | Symmetric (40%) | 66.86 ± 0.35     | 64.80 ± 0.40           |

### Fashion-MNIST Results

| Noise Type      | SGN Accuracy (%) | Our Implementation (%) |
|-----------------|------------------|-------------------------|
| No Noise (0%)   | 91.05 ± 0.30     | 89.50 ± 0.40           |
| Symmetric (20%) | 88.90 ± 0.25     | 87.20 ± 0.30           |
| Symmetric (40%) | 85.50 ± 0.40     | 83.60 ± 0.45           |
| Symmetric (60%) | 80.20 ± 0.60     | 78.00 ± 0.70           |
| Asymmetric (20%)| 89.00 ± 0.35     | 87.50 ± 0.40           |
| Asymmetric (40%)| 85.80 ± 0.50     | 84.50 ± 0.55           |

---

## Applications

## Applications

The robust learning techniques proposed in this project have wide-ranging applications across various domains:

### 1. **Medical Diagnosis**
- Robust learning can enhance classification accuracy in medical imaging (e.g., X-rays, MRIs), where human errors may introduce noisy labels.
- Applications include automated disease detection and diagnosis.

### 2. **Autonomous Vehicles**
- Real-world driving datasets often contain noisy annotations due to occlusions, lighting conditions, and sensor errors.
- Robust methods improve the reliability of safety-critical applications, such as pedestrian detection and lane identification.

### 3. **E-Commerce**
- Product categorization on e-commerce platforms frequently suffers from mislabeled or incomplete data.
- Robust classification can streamline recommendation systems and inventory management.

### 4. **Fraud Detection**
- Financial datasets often have mislabeled or ambiguous data entries.
- The proposed regression-based framework can improve fraud detection systems by addressing data inconsistencies.

### 5. **Natural Language Processing**
- Text classification tasks, such as sentiment analysis or spam detection, can benefit from robust learning methods to mitigate noisy labels in large-scale datasets.

By leveraging the strengths of regression-based classification and noise handling, these applications can achieve greater accuracy, reliability, and interpretability.
