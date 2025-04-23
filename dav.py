import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Re-generate the synthetic dataset to produce image outputs again
regions = [
    'East Midlands', 'East of England', 'London', 'North East',
    'North West', 'South East', 'South West', 'West Midlands',
    'Yorkshire and the Humber'
]
disease_types = ['H5N1', 'H5N8', 'NDV', 'IBV']
animal_types = ['Chicken', 'Duck', 'Wild Bird']
case_statuses = ['Confirmed', 'Suspected']
date_range = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
num_samples = 950

# Create synthetic dataset
import numpy as np
np.random.seed(42)
data = {
    'date': np.random.choice(date_range, size=num_samples),
    'region': np.random.choice(regions, size=num_samples),
    'disease_type': np.random.choice(disease_types, size=num_samples),
    'animal_type': np.random.choice(animal_types, size=num_samples),
    'case_status': np.random.choice(case_statuses, size=num_samples),
    'case_count': np.random.poisson(lam=3, size=num_samples),
    'temperature': np.random.normal(loc=10, scale=5, size=num_samples).round(1),
    'humidity': np.random.uniform(60, 95, size=num_samples).round(1),
    'rainfall': np.random.exponential(scale=3.0, size=num_samples).round(1),
    'wind_speed': np.random.normal(loc=15, scale=5, size=num_samples).round(1),
}
df = pd.DataFrame(data)
df['case_count'] = df['case_count'].apply(lambda x: max(x, 0))

# Store image paths
sns.set(style="whitegrid")
image_outputs = {}

# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df['case_count'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Case Counts")
plt.xlabel("Number of Cases")
plt.ylabel("Frequency")
hist_path = "/mnt/data/eda_hist_case_count.png"
plt.savefig(hist_path)
image_outputs['Histogram'] = hist_path
plt.close()

# Scatter plots
env_vars = ['temperature', 'humidity', 'rainfall', 'wind_speed']
for var in env_vars:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=var, y='case_count', alpha=0.6)
    plt.title(f"Case Count vs {var.title()}")
    plt.xlabel(var.title())
    plt.ylabel("Case Count")
    path = f"/mnt/data/eda_scatter_{var}.png"
    plt.savefig(path)
    image_outputs[f"Scatter_{var}"] = path
    plt.close()

# Box plots
cat_vars = ['animal_type', 'disease_type', 'region', 'case_status']
for cat in cat_vars:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x=cat, y='case_count')
    plt.title(f"Case Count by {cat.replace('_', ' ').title()}")
    plt.xlabel(cat.replace('_', ' ').title())
    plt.ylabel("Case Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = f"/mnt/data/eda_box_{cat}.png"
    plt.savefig(path)
    image_outputs[f"Box_{cat}"] = path
    plt.close()

# Heatmap
plt.figure(figsize=(8, 6))
numerical_vars = ['case_count', 'temperature', 'humidity', 'rainfall', 'wind_speed']
corr = df[numerical_vars].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
heatmap_path = "/mnt/data/eda_corr_heatmap.png"
plt.savefig(heatmap_path)
image_outputs["Correlation_Heatmap"] = heatmap_path
plt.close()

image_outputs
