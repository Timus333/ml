#ml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {'Area': [3100, 2700, 2600, 4000, 3300, 2100], 'Price': [6200000, 5800000, 5400000, 7500000, 6500000, 5500000]}
df = pd.DataFrame(data)
X = df[['Area']] 
y = df['Price']
model = LinearRegression()   
model.fit(X, y)
new_areas = np.array([3500, 5000]).reshape(-1, 1)
new_areas_df = pd.DataFrame(new_areas, columns=['Area'])
predicted_prices = model.predict(new_areas_df)
for area, price in zip(new_areas.flatten(), predicted_prices):
    print(f"Area: {area} sq.ft -> Predicted Price: Rs {price:.2f}")
rmse = np.sqrt(mean_squared_error(y, model.predict(X)))
r2 = r2_score(y, model.predict(X))
print(f"\nRMSE: {rmse:.2f}, R²: {r2:.2f}")
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.scatter(new_areas, predicted_prices, color='green', label='Predictions')
plt.xlabel('Area (sq.ft)')
plt.ylabel('Price (Rs)')
plt.title('Home Prices Prediction')
plt.legend()
plt.show()
