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

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('https://raw.githubusercontent.com/Prof-Nirbhay/Machine-Learning/refs/heads/main/GolfPlay.csv')
label_encoder = LabelEncoder()
for col in df.columns:
    df[col] = label_encoder.fit_transform(df[col])
X, y = df.drop('PlayGolf', axis=1), df['PlayGolf']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
url = "https://raw.githubusercontent.com/Prof-Nirbhay/Machine-Learning/main/GolfPlay.csv"
data = pd.read_csv(url)
data = pd.get_dummies(data, drop_first=True)
X, y = data.drop('PlayGolf_Yes', axis=1), data['PlayGolf_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
plt.figure(figsize=(20, 10))
plot_tree(rf_model.estimators_[0], feature_names=X.columns, class_names=["No", "Yes"], filled=True, rounded=True, fontsize=8)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
url = "https://raw.githubusercontent.com/Prof-Nirbhay/Machine-Learning/refs/heads/main/Sale_data.csv"
df = pd.read_csv(url)
X, y = df[['Age', 'Income', 'Purchase_History']], df['Purchase_Decision']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
error_rate = [np.mean(KNeighborsClassifier(n_neighbors=i).fit(X_train_scaled, y_train).predict(X_test_scaled) != y_test) 
for i in range(1, 21)]
plt.figure(figsize=(8, 5))
plt.plot(range(1, 21), error_rate, marker='o', linestyle='dashed', color='blue')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.title('Error Rate vs. K')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
url = "https://raw.githubusercontent.com/Prof-Nirbhay/Machine-Learning/main/logistic_data.csv"
df = pd.read_csv(url)
X, y = df.drop(columns=["Label"]), df["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
y_pred = log_reg.predict(X_test_scaled)
print("Logistic Regression Performance:\n", classification_report(y_test, y_pred))
plt.figure(figsize=(8, 6))
plt.scatter(X_test["Feature1"], X_test["Feature2"], c=y_test, cmap="coolwarm", edgecolors="k")
plt.xlabel("Feature1"), plt.ylabel("Feature2")
plt.title("Logistic Regression Model Plot")
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
X, y = make_blobs(n_samples=500, centers=4, random_state=42)
X_scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')  # Centers
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
coefficients = pd.Series(lasso.coef_, index=X.columns)
important_features = coefficients[coefficients != 0]
print("All Coefficients:\n", coefficients.sort_values(ascending=False))
print("\nImportant Features:\n", important_features.sort_values(ascending=False))
