
from mlp import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# URL for Auto MPG dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
df = pd.read_csv(url, sep='\\s+', names=columns, na_values="?")

# Drop 'car_name' since it's not needed for numerical analysis
df.drop(columns=["car_name"], inplace=True)

# Drop missing values
df.dropna(inplace=True)

# Ensure 'origin' is an integer
df["origin"] = df["origin"].astype(int)

# Convert categorical column ('origin') to numerical (one-hot encoding)
df = pd.get_dummies(df, columns=['origin'], drop_first=True)

# Separate features and target variable
X = df.drop(columns=["mpg"])
y = df["mpg"].values.reshape(-1, 1)

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training (70%), validation (15%), and testing (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)


mlp = MultilayerPerceptron([
    Layer(fan_in=X_train.shape[1], fan_out=64, activation_function=Relu()),
    Layer(fan_in=64, fan_out=32, activation_function=Relu(),dropout_rate=0.07),
    Layer(fan_in=32, fan_out=1, activation_function=Linear())  
])


loss_func = SquaredError()

training_losses, validation_losses = mlp.train(
    train_x=X_train, train_y=y_train,
    val_x=X_test, val_y=y_test,
    loss_func=loss_func,
    learning_rate=0.001, batch_size=64, epochs=80,
    rmsprop=True, 
    beta=0.9,      
    epsilon=1e-8   
)

from sklearn.metrics import mean_squared_error, r2_score

y_pred = mlp.forward(X_test, training=False)

y_pred = y_pred.flatten()
y_test = y_test.flatten()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(training_losses, label="Training Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()

# Plot validation loss
plt.subplot(1, 2, 2)
plt.plot(validation_losses, label="Validation Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Validation Loss Over Time")
plt.legend()

plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Select 10 random samples from the test set
num_samples = 10
indices = np.random.choice(len(X_test), num_samples, replace=False)
sample_X = X_test[indices]
sample_y_true = y_test[indices].flatten()  # Ensure 1D array
sample_y_pred = mlp.forward(sample_X, training=False).flatten()  # Flatten predictions

# Compute error metrics
absolute_errors = np.abs(sample_y_true - sample_y_pred)

results_df = pd.DataFrame({
    "Sample Index": indices,
    "True MPG": sample_y_true,
    "Predicted MPG": sample_y_pred,
    "Absolute Error": absolute_errors
})

# Save results to CSV
results_df.to_csv("predicted_vs_true_mpg.csv", index=False)
print("Saved predicted MPG results to 'predicted_vs_true_mpg.csv'")

# Display the table using PrettyTable
table = PrettyTable()
table.field_names = ["Index", "True MPG", "Predicted MPG", "Abs Error"]

for _, row in results_df.iterrows():
    table.add_row([row["Sample Index"], f"{row['True MPG']:.2f}", f"{row['Predicted MPG']:.2f}", f"{row['Absolute Error']:.2f}"])

print("\n" + "="*50)
print("🔹 Predicted vs True MPG (10 Random Samples)")
print("="*50)
print(table)
print("="*50)



# Scatter Plot of True vs Predicted MPG
plt.figure(figsize=(8, 6))
plt.scatter(sample_y_true, sample_y_pred, color="blue", label="Predictions")
plt.plot([min(sample_y_true), max(sample_y_true)], [min(sample_y_true), max(sample_y_true)], "r--", label="Perfect Prediction")
plt.xlabel("True MPG")
plt.ylabel("Predicted MPG")
plt.title("True vs Predicted MPG for 10 Random Samples")
plt.legend()
plt.show()
