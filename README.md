# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:

### Register Number:

```python
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

df = pd.read_csv('student_study_hours_regression.csv')

X = torch.tensor(df[['Study_Hours']].values, dtype=torch.float32)
y = torch.tensor(df[['Exam_Marks']].values, dtype=torch.float32)

print("Input Shape:", X.shape)
print("Output Shape:", y.shape)

torch.manual_seed(59)

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

torch.manual_seed(59)
model = Model(1, 1)

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

epochs = 300
losses = []

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_function(y_pred, y)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

plt.figure(figsize=(10, 6))
plt.plot(range(epochs), losses, linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error Loss')
plt.title('Experiment No. 1 - Regression Training Loss')
plt.grid(True, alpha=0.3)

plt.figtext(
    0.5, 0.015,
    'Name: AKSHAY KARTHICK ASR  |  Reg. No.: 212224230015',
    ha='center',
    fontsize=11
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

w1 = model.linear.weight.item()
b1 = model.linear.bias.item()

print("\n========== TRAINED MODEL RESULTS ==========")
print(f"Final Weight: {w1:.8f}")
print(f"Final Bias: {b1:.8f}")
print(f"Final Training Loss: {losses[-1]:.8f}")

x1 = torch.tensor(
    [[X.min().item()], [X.max().item()]],
    dtype=torch.float32
)

with torch.no_grad():
    y1 = model(x1)

plt.figure(figsize=(10, 6))

plt.scatter(
    X.numpy(),
    y.numpy(),
    color='red',
    label='Original Data'
)

plt.plot(
    x1.numpy(),
    y1.numpy(),
    color='blue',
    label='Best-Fit Line'
)

plt.xlabel('Study Hours')
plt.ylabel('Exam Marks')
plt.title('Experiment No. 1 - Trained Regression Model')
plt.legend()
plt.grid(True, alpha=0.3)

plt.figtext(
    0.5, 0.015,
    'Name: AKSHAY KARTHICK ASR  |  Reg. No.: 212224230015',
    ha='center',
    fontsize=11
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()

```

### OUTPUT
Training Loss Vs Iteration Plot

<img width="1047" height="602" alt="Screenshot 2026-10-21 at 1 48 46 PM" src="https://github.com/user-attachments/assets/9c1065e3-932b-487b-9e5b-8b478302799c" />


Best Fit line plot

<img width="1045" height="596" alt="Screenshot 2026-10-21 at 1 49 14 PM" src="https://github.com/user-attachments/assets/2f52d490-b0ec-4c49-b021-3f66d5bee91f" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
