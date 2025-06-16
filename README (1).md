
# 🧠 Digit Recognizer using Neural Network from Scratch

This repository implements a handwritten digit recognizer built entirely from scratch using NumPy. It trains on the [MNIST digit dataset](https://www.kaggle.com/competitions/digit-recognizer/) and predicts digits 0–9 from 28x28 grayscale images, without using any high-level deep learning libraries like TensorFlow or PyTorch.

---

## 🔍 Overview

- **Architecture:** 2-layer fully connected neural network
- **Input Layer:** 784 units (28x28 pixels flattened)
- **Hidden Layer:** 10 units with ReLU activation
- **Output Layer:** 10 units with Softmax activation (for digit classes 0–9)
- **Loss Function:** Cross-entropy
- **Optimizer:** Gradient Descent

---

## 🧮 Mathematical Formulation

### 🔁 Forward Propagation

![Forward Propagation](./images/Screenshot_2025-06-16_174841.png)

### 🔄 Backward Propagation & Parameter Updates

Follows standard backpropagation with partial derivatives for updating weights and biases using the gradients of loss w.r.t each parameter.

---

## 📐 Variable Shapes

Shapes for all major variables in forward and backward propagation:

![Variable Shapes](./images/Screenshot_2025-06-16_174852.png)

---

## 📂 Dataset

This uses the [Kaggle Digit Recognizer dataset](https://www.kaggle.com/competitions/digit-recognizer/):

- Each example is a 28x28 pixel grayscale image.
- Labels are digits `0–9`.
- Dataset is pre-split into training and dev sets.

---

## 🧰 Project Structure

```
.
├── digit_recognizer.py     # All logic for model, training, testing
├── train.csv               # Dataset (from Kaggle)
├── images/                 # Diagrams for forward/backward propagation
│   ├── Screenshot_2025-06-16_174841.png
│   └── Screenshot_2025-06-16_174852.png
├── README.md               # This file
```

---

## 🚀 How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/).
2. Place `train.csv` in the same directory as the code.
3. Run the training and prediction:
```bash
python digit_recognizer.py
```

---

## 🧪 Sample Predictions

Test individual digits using:
```python
test_prediction(index, W1, b1, W2, b2)
```

You’ll get the prediction, label, and a visual of the digit image.

---

## ✍️ Implementation Details

- Manual matrix multiplication (no `Dense` layers)
- One-hot encoding for labels
- ReLU and Softmax activation functions
- Gradient descent with fixed learning rate
- Modular code: separate functions for forward, backward, update, and prediction

---

## 📊 Performance

The model achieves decent accuracy (~85–90%) on MNIST with just 500 iterations and basic architecture.

---

## 📌 TODO

- Add cross-validation
- Add regularization (L2, dropout)
- Visualize loss over epochs
- Save/load model parameters

---

## 🧠 Concepts Used

- Neural networks
- Activation functions
- Backpropagation
- Chain rule of derivatives
- One-hot encoding
- Numpy broadcasting

---



---

Feel free to fork or contribute to make it even better!
