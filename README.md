# Structural Integrity Monitor (SHM System)

An automated, deep-learning-based Structural Health Monitoring (SHM) system engineered to detect surface fatigue, fractures, and material defects across maritime hulls and industrial infrastructure using computer vision.

## Project Performance Metrics
* **Dataset Scale:** 39,836 high-resolution industrial image samples
* **Data Split:** Systematic partitioning into `Positive` (Defective) and `Negative` (Intact) structural profiles
* **Final Training Accuracy:** **99.66%**
* **Final Test Accuracy (Unseen Data):** **99.42%**

---

## Tech Stack & Architecture
* **Framework:** TensorFlow 2.x / Keras
* **Core Libraries:** NumPy, OpenCV, Matplotlib
* **Model Topology:** Custom Deep Convolutional Neural Network (CNN)
  * `Rescaling` Input Layer (150 x 150 x 3, normalized to [0, 1])
  * `Conv2D` Spatial Feature Extraction (32 filters, 3 x 3 kernel, ReLU activation)
  * `MaxPooling2D` Dimensionality Reduction (2 x 2 pool size)
  * `Flatten` Layer for dense vector translation
  * `Dense` Hidden Layer (64 units, ReLU activation)
  * `Dense` Output Layer (1 unit, Sigmoid activation for binary classification)
* **Compilation Pipeline:** Adam Optimizer paired with Binary Cross-Entropy loss computation

---

## Repository Structure
```text
structural-integrity-monitor/
├── data/
│   ├── train/          # 39,836 training images (Positive/Negative)
│   └── test/           # Unseen evaluation images (Positive/Negative)
├── model/
│   └── safety_model.keras  # Extracted production-ready trained weights
└── src/
    ├── train_model.py  # Optimized iterative model compilation script
    ├── validate.py     # Test-set evaluation performance script
    └── predict.py      # Live inference image classification utility
