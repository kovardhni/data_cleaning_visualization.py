
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")
print("First 5 Rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)
df.drop_duplicates(inplace=True)
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("\nCleaned Dataset Shape:", df.shape)
df.to_csv("cleaned_data.csv", index=False)
plt.figure(figsize=(8,5))
df[numeric_cols[0]].hist(bins=20)
plt.title("Distribution of First Numeric Column")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(data=df[numeric_cols])
plt.title("Box Plot")
plt.xticks(rotation=45)
plt.show()
sns.pairplot(df[numeric_cols])
plt.show()

print("Data Cleaning and Visualization Completed Successfully!")