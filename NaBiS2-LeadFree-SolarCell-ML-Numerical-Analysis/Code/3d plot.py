import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("etl_data.csv")
# Remove rows with all NaNs or headers repeated
df = df.dropna(how='all')
df = df[df.columns[1:]].apply(pd.to_numeric, errors='coerce')  # convert to numeric

# Drop remaining NaNs
df.dropna(inplace=True)

# Example Visualization
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x=" x(um)", y=" jtot(mA/cm2)")
plt.title("Total Current Density vs Position")
plt.xlabel("Position (um)")
plt.ylabel("Total Current Density (mA/cm²)")
plt.grid(True)
plt.show()
