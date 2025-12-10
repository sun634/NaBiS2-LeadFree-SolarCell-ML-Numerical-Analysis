# Create an extended ML script with multiple models: RandomForest, DecisionTree, and XGBoost (if installed)

ml_script_extended = """# NaBiS2 Solar Cell ML Comparison (Multiple Models)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('shunt_resistance.csv')

# Features and Target
X = df.drop(columns=['PCE'])
y = df['PCE']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42)
}

try:
    from xgboost import XGBRegressor
    models['XGBoost'] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
except ImportError:
    print("⚠️ XGBoost is not installed. Skipping XGBoost model.")

# Train and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'R2': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'Feature Importances': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    }
    print(f"\\nModel: {name}")
    print("R2 Score:", results[name]['R2'])
    print("MSE:", results[name]['MSE'])

# Plot Feature Importances (for models that support it)
for name in results:
    importance = results[name]['Feature Importances']
    if importance is not None:
        plt.figure(figsize=(6,4))
        plt.barh(X.columns, importance, color='coral')
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance: {name}')
        plt.tight_layout()
        filename = f'feature_importance_{name.replace(" ", "_").lower()}.png'
        plt.savefig(filename)
        plt.show()
"""

py_extended_path = "/mnt/data/nabis2_ml_comparison.py"
with open(py_extended_path, "w") as f:
    f.write(ml_script_extended)

py_extended_path
