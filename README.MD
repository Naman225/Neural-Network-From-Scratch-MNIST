# 🧠 Neural Network From Scratch (MNIST)

A fully vectorized implementation of a **multi-layer neural network built from scratch using NumPy**, trained on the MNIST dataset for handwritten digit classification.

This project demonstrates a deep understanding of **forward propagation, backpropagation, and multi-class classification** without using deep learning frameworks like TensorFlow or PyTorch.

---

## 🚀 Key Features

* Neural network implemented **from scratch (NumPy only)**
* Multi-layer architecture with **ReLU + Softmax**
* **Categorical Cross Entropy Loss**
* Efficient **vectorized forward & backward propagation**
* Gradient descent optimization
* Model evaluation using:

  * Accuracy
  * Confusion Matrix
* Error analysis via **wrong prediction inspection**

---

## 📊 Results

* **Test Accuracy:** ~96–97%
* Stable training convergence
* Identified common misclassifications:

  * 5 ↔ 3
  * 4 ↔ 9
  * 7 ↔ 2

---

## 🏗️ Project Structure

```
src/
│
├── train.py                 # Training loop
├── model.py                 # Neural network (forward + backward)
├── evaluate.py              # Metrics & confusion matrix
├── data_ingestion.py        # Data loading
├── data_transformation.py   # Preprocessing & encoding
```

---

## 🧠 Model Architecture

```
Input Layer:      784 (28×28 image)
Hidden Layer 1:   ReLU
Hidden Layer 2:   ReLU
Output Layer:     10 (Softmax)
```

---

## ⚙️ Training Details

* Loss Function: **Categorical Cross Entropy**
* Optimization: **Gradient Descent**
* Backprop Simplification:

  ```
  dZ = AL - Y
  ```

---

## 📦 Dataset

The dataset is not included due to GitHub size limits.

Download MNIST (CSV format) from:
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

Place the files in:

```
data/raw/
```

---

## ▶️ How to Run

```bash
git clone https://github.com/Naman225/Neural-Network-From-Scratch-MNIST.git
cd Neural-Network-From-Scratch-MNIST

pip install -r requirements.txt
python src/train.py
```

---

## 📈 Future Improvements

* Mini-batch gradient descent
* Learning rate tuning / scheduling
* Deeper architectures
* Convolutional Neural Networks (CNNs)
* Visualization of training curves

---

## 💡 Key Learnings

* Built a neural network pipeline from scratch
* Understood and implemented backpropagation mathematically
* Debugged model using confusion matrix and error analysis
* Improved performance through systematic experimentation

---

## 👨‍💻 Author

**Naman Tiwari**

---

## ⭐ If you found this useful

Consider giving the repository a star ⭐
