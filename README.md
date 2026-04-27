# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Import required libraries and create dataset with input features (area, bedrooms) and outputs (price, occupants).
2.  Separate features (X) and target variables (y) from the dataset.
3.  Apply feature scaling to both input and output data using StandardScaler.
4.  Initialize MultiOutputRegressor with SGDRegressor and train the model using scaled data.
5.  Predict outputs using the trained model and convert them back to original scale.
6. Plot actual vs predicted values and use the model to predict price and occupants for new input. 
## program:
# Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
# Developed by: VIJIYALAKSHMI A
# RegisterNumber: 212225240185 
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# Step 1: Dataset (Area, Bedrooms → Price, Occupants)
data = {
    "Area": [900, 1200, 1500, 1800, 2100, 2500],
    "Bedrooms": [2, 2, 3, 3, 4, 4],
    "Price": [180000, 240000, 300000, 360000, 420000, 500000],
    "Occupants": [3, 4, 5, 5, 6, 7]
}

df = pd.DataFrame(data)

# Step 2: Features & Targets
X = df[["Area", "Bedrooms"]]
y = df[["Price", "Occupants"]]

# Step 3: Scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Step 4: Model (Multi-output SGD)
base_model = SGDRegressor(max_iter=1500, eta0=0.01, learning_rate='constant')
model = MultiOutputRegressor(base_model)

# Step 5: Train
model.fit(X_scaled, y_scaled)

# Step 6: Prediction
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Step 7: Scatter Plot (Actual vs Predicted Price)
plt.figure()
plt.scatter(y["Price"], y_pred[:, 0])

# Perfect line
plt.plot([y["Price"].min(), y["Price"].max()],
         [y["Price"].min(), y["Price"].max()])

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price (SGD Multi-output)")
plt.grid()
plt.show()

# Step 8: New Prediction
new_house = [[2000, 3]]
new_scaled = scaler_X.transform(new_house)

pred_scaled = model.predict(new_scaled)
pred = scaler_y.inverse_transform(pred_scaled)

print("Predicted Price:", pred[0][0])
print("Predicted Occupants:", pred[0][1])
```
## Output:
<img width="865" height="595" alt="Screenshot 2026-04-27 142623" src="https://github.com/user-attachments/assets/e8330e4a-e840-4c73-8147-456dc86ed001" />
<img width="1714" height="107" alt="Screenshot 2026-04-27 142633" src="https://github.com/user-attachments/assets/70776c13-3ba1-4819-b8b5-bda29a026a74" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
