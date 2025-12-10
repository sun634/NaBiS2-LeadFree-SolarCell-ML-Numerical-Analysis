import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and clean the data
df = pd.read_csv('shunt_resistance.csv')  # CSV path
df.columns = [col.strip().replace(" ", "_").replace("(", "").replace(")", "") for col in df.columns]
df.dropna(inplace=True)

# 2. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# 3. Scatter Matrix
sns.pairplot(df)
plt.savefig('scatter_matrix.png')
plt.close()

# 4. Feature selection & scaling
X = df.drop(columns=['PCE'])  # Predicting Power Conversion Efficiency
y = df['PCE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. ANN Model
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu',
                     solver='adam', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# 6. Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"ANN Model Results:")
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# 7. Prediction Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color='teal', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual PCE')
plt.ylabel('Predicted PCE')
plt.title('Actual vs Predicted PCE (ANN)')
plt.tight_layout()
plt.savefig('pce_prediction_ann.png')
plt.close()
