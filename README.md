# 🌌 Calolib: CaloGAN + CaloDiff

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Training_&_Inference-blue)

**Calolib** is a library and research framework that serves as a continuation of the **CaloGAN** project. It is dedicated to the generative modeling (simulation) of particle showers in the Large Hadron Collider (LHC) calorimeters using state-of-the-art diffusion models.

This repository contains the key practical materials and source code developed as part of a Master's thesis.

---

## 📑 Table of Contents
1. [About the Calolib Framework](#-about-the-calolib-framework)
2. [Evaluation and Validation](#-evaluation-and-validation)
3. [Training and Inference](#-training-and-inference)

---

## 🛠 About the Calolib Framework

The core codebase of the framework is located in the pipeline directories and contains all the necessary tools for working with diffusion models:

* **`pipeline/calodiff.py`** — The core of the library. It contains the implementation of the diffusion model and advanced methods for its acceleration (fast sampling):
    * Model Distillation
    * Optimized Solvers
    * Caching Mechanisms
* **`pipeline/physical_metrics/`** — A module containing the implementation of specialized physical metrics to evaluate the quality of the generated particle showers.
* **`vizualization_aux/`** — A set of auxiliary methods and utilities for clear visualization of generation results and physical distributions.

---

## 📊 Evaluation and Validation

The repository includes Jupyter notebooks with detailed analysis, metric testing, and statistical justification of the results:

| Notebook | Description |
| :--- | :--- |
| 📓 `Validating Models With Different Complexity.ipynb` | Verification and comparison of models with varying architectural complexity. |
| 📓 `Evaluating models with different complexity and symmetry schedule for caching.ipynb` | Testing models of varying complexity using an optimized (symmetric) schedule of diffusion steps for caching. |
| 📓 `MathStat Validating result.ipynb` | Rigorous mathematical and statistical justification of the stability and performance of the simplest (lightweight) model relative to the baseline. |

---

## 🚀 Training and Inference

Due to high computational requirements, the training and inference of heavy models were conducted in the Kaggle cloud environment. The code and run history are available at the following links:

### 🏋️‍♂️ Training
* **Link:** [alex55555/calodiff-training](https://www.kaggle.com/code/alex55555/calodiff-training)
* **UNet Architecture:** Versions `3`, `12-15`
* **DiT (Diffusion Transformer) Architecture:** Versions `8-11`

### 🧠 Inference
* **Link:** [alex55555/calodiff-inference](https://www.kaggle.com/code/alex55555/calodiff-inference)
* **Description:** The notebook used for generating samples with heavy models and measuring sampling speed and quality.

---

> **Note:** This repository is part of a master's research project. If you have any questions regarding the implementation or physical metrics, please feel free to open an *Issue* in this repository.
