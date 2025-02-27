# 🚀 How to Run

Ensure you have installed the required dependencies:

```bash
pip install numpy pandas scikit-learn torch torchvision matplotlib prettytable
```

## 🏃 Running the Models

### **1️⃣ Train the MNIST Classification Model**
Run the following command to train the MLP on the MNIST dataset:

```bash
python train_mnist.py
```

### **2️⃣ Train the Auto MPG Regression Model**
To train the MLP for vehicle fuel efficiency prediction, use:

```bash
python train_mpg.py
```

## 📊 Viewing Results
- The MNIST model outputs accuracy, a classification report, and visualizations.
- The Auto MPG model outputs Mean Squared Error (MSE), R² score, and a comparison of predicted vs. actual MPG values.

For troubleshooting or modifications, check `mlp.py` for the model architecture.
