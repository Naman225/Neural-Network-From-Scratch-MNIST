🧠 Neural Network From Scratch (MNIST)

A fully functional multi-layer neural network built from scratch using NumPy, trained on the MNIST dataset for handwritten digit classification.

This project demonstrates a deep understanding of forward propagation, backpropagation, and multi-class classification without using any deep learning frameworks like TensorFlow or PyTorch.

🚀 Features
✅ Neural network implemented from scratch (NumPy only)
✅ Multi-layer architecture with ReLU + Softmax
✅ Categorical Cross Entropy Loss
✅ Efficient vectorized forward & backward propagation
✅ Gradient descent optimization
✅ Custom training pipeline
✅ Model evaluation with:
Accuracy
Confusion Matrix
✅ Error analysis using wrong predictions visualization
📊 Results
🎯 Accuracy: ~96–97% on MNIST test set
📉 Stable loss convergence
🔍 Identified common confusions:
5 ↔ 3
4 ↔ 9
7 ↔ 2
🏗️ Project Structure
src/
│
├── train.py              # Training pipeline
├── model.py              # Neural network implementation
├── evaluate.py           # Accuracy & confusion matrix
├── data_ingestion.py     # Dataset loading
├── data_transformation.py # Preprocessing & one-hot encoding
🧠 Model Architecture
Input Layer: 784 (28x28 pixels)
Hidden Layer 1: ReLU
Hidden Layer 2: ReLU
Output Layer: 10 (Softmax)
⚙️ How It Works
Forward Propagation
Linear → ReLU (hidden layers)
Linear → Softmax (output layer)
Loss Function
Categorical Cross Entropy
Backpropagation

Optimized gradient:

dZ = AL - Y
Prediction
Uses argmax to select class with highest probability
📦 Dataset
MNIST handwritten digits dataset
60,000 training samples
10,000 test samples
▶️ How to Run
git clone https://github.com/your-username/Neural-Network-From-Scratch-MNIST.git
cd Neural-Network-From-Scratch-MNIST
python train.py
📈 Future Improvements
🔹 Mini-batch gradient descent
🔹 Learning rate scheduling
🔹 Deeper architectures
🔹 CNN implementation
🔹 Visualization of training curves
💡 Key Learnings
Implemented full deep learning pipeline without frameworks
Understood backpropagation mathematically
Debugged real-world model errors using confusion matrix
Improved model performance through error analysis
👨‍💻 Author

Naman Tiwari