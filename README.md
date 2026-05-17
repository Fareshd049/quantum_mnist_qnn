# Quantum Image Classification with TensorFlow Quantum

A binary image classifier built on a **Quantum Neural Network (QNN)** using TensorFlow Quantum and Cirq. The model classifies Fashion MNIST images into two categories — T-shirts/tops (class 0) vs. Shirts (class 6) — by encoding pixel data as quantum circuits and training a Parameterized Quantum Circuit (PQC).

---

## Overview

This notebook explores a hybrid classical-quantum approach to image classification:

1. Images are downscaled, binarized, and encoded as quantum circuits using Cirq.
2. A QNN with XX and ZZ entangling layers processes the circuits.
3. TensorFlow Quantum's `PQC` layer trains the model end-to-end using hinge loss.

---

## Requirements

| Package | Version |
|---|---|
| TensorFlow | 2.15.1 |
| TensorFlow Quantum | 0.7.5 |
| NumPy | 1.26.4 |
| SciPy | 1.12.0 |
| Cirq | latest |
| Protobuf | 4.25.3 |
| scikit-learn | latest |

Install dependencies:

```bash
pip install tensorflow==2.15.1 tensorflow-quantum==0.7.5 numpy==1.26.4 scipy==1.12.0 protobuf==4.25.3 cirq scikit-learn
```

---

## Pipeline

### 1. Data Loading & Filtering
- Loads the **Fashion MNIST** dataset via `tf.keras.datasets`.
- Filters to keep only classes **0** (T-shirt/top) and **6** (Shirt) for binary classification.

### 2. Preprocessing
- Normalizes pixel values to `[0, 1]`.
- Resizes images from **28×28** down to **2×2** using TensorFlow's `tf.image.resize`.
- Splits into train / validation (90/10) / test sets.

### 3. Quantum Encoding
- Flattens each 2×2 image into 4 pixel values.
- Applies **binary encoding** (threshold = 0.5): pixels above threshold → `1`, below → `0`.
- Constructs a **Cirq quantum circuit** per image: active pixels apply an `X` gate to the corresponding qubit.
- Converts circuits to TFQ tensors via `tfq.convert_to_tensor`.

### 4. QNN Architecture
- **Data qubits**: a 2×2 grid of `cirq.GridQubit`s.
- **Readout qubit**: an ancilla qubit initialized with `X` and `H` gates.
- **Layers**: parameterized `XX` and `ZZ` entangling gates across all data-readout qubit pairs (symbols learned during training).
- **Measurement**: `cirq.Z` on the readout qubit.

### 5. Training
- Model: `tf.keras.Sequential` with a single `tfq.layers.PQC` layer.
- Loss: `tf.keras.losses.Hinge()`
- Optimizer: `Adam(lr=0.001)`
- Metric: custom `hinge_accuracy`
- Trained for **10 epochs**, batch size **64**.

---

## Usage

Run all cells in order in a Jupyter environment (Google Colab recommended due to TFQ installation requirements):

```
Image_Classification.ipynb
```

Model weights are saved after training:

```python
model.save_weights('content/')
```

---

## Notes

- TensorFlow Quantum requires specific TF versions. The notebook reinstalls `tensorflow==2.15.1` mid-way to ensure compatibility with `tensorflow-quantum`.
- This notebook is best run on **Google Colab** where GPU/TPU support and package management are straightforward.
- The binary classification task (class 0 vs class 6) and aggressive downscaling (28×28 → 2×2) are necessary constraints due to the limited qubit count in near-term quantum devices.
