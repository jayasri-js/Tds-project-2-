pip install matplotlib
pip install seaborn


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_analyze_dataset(dataset_path):
    encodings = ['utf-8', 'utf-16', 'ISO-8859-1']
    df = None

    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)

    for encoding in encodings:
        try:
            df = pd.read_csv(dataset_path, encoding=encoding)
            print(f"Dataset loaded successfully with {encoding} encoding!")
            break
        except UnicodeDecodeError:
            print(f"Error with {encoding} encoding. Trying next encoding...")
        except Exception as e:
            print(f"Error loading dataset with {encoding} encoding: {e}")

    if df is None:
        print("Failed to load the dataset with all attempted encodings.")
        return

    # Basic dataset statistics
    summary = df.describe()
    missing_values = df.isnull().sum()
    skewness = df.select_dtypes(include=['number']).skew()
    kurtosis = df.select_dtypes(include=['number']).kurtosis()

    # Heatmap of missing values
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    heatmap_path = os.path.join(output_dir, "missing_values_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()

    # Histograms for numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    histogram_paths = []
    if len(numeric_columns) > 0:
        for column in numeric_columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[column], kde=True, bins=30)
            plt.title(f'{column} Distribution')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            histogram_path = os.path.join(output_dir, f"{column}_distribution.png")
            plt.savefig(histogram_path)
            plt.close()
            histogram_paths.append(histogram_path)

    # Correlation matrix
    correlation_matrix = df[numeric_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    correlation_matrix_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(correlation_matrix_path)
    plt.close()

    # Outlier detection
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).sum()

    # Generate insights locally
    insights = generate_local_insights(missing_values, skewness, kurtosis, correlation_matrix, outliers)

    # Create README file
    create_readme(output_dir, missing_values, skewness, kurtosis, correlation_matrix_path, histogram_paths, insights)

def generate_local_insights(missing_values, skewness, kurtosis, correlation_matrix, outliers):
    # Summarize findings
    insights = []
    insights.append("### Insights\n")
    insights.append("1. **Missing Values:**\n")
    insights.append(missing_values.to_string() + "\n")
    insights.append("\n2. **Skewness:**\n")
    insights.append(skewness.to_string() + "\n")
    insights.append("\n3. **Kurtosis:**\n")
    insights.append(kurtosis.to_string() + "\n")
    insights.append("\n4. **Correlation Matrix:**\n")
    insights.append(correlation_matrix.to_string() + "\n")
    insights.append("\n5. **Outliers:**\n")
    insights.append(outliers.to_string() + "\n")

    # Add observations based on the analysis
    insights.append("\n### Observations:\n")
    if missing_values.sum() > 0:
        insights.append("- The dataset contains missing values. Consider imputing or removing them.\n")
    else:
        insights.append("- No missing values detected.\n")

    if skewness.abs().max() > 1:
        insights.append("- Some numeric features exhibit high skewness. Consider transformations like log or square root.\n")

    if kurtosis.abs().max() > 3:
        insights.append("- Some features have extreme kurtosis, indicating heavy tails or outliers.\n")

    return "\n".join(insights)

def create_readme(output_dir, missing_values, skewness, kurtosis, correlation_matrix_path, histogram_paths, insights):
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Dataset Analysis Report\n\n")
        f.write("## Missing Values\n")
        f.write(missing_values.to_string() + "\n\n")
        f.write("## Skewness of Numeric Columns\n")
        f.write(skewness.to_string() + "\n\n")
        f.write("## Kurtosis of Numeric Columns\n")
        f.write(kurtosis.to_string() + "\n\n")
        f.write("## Visualizations\n")
        f.write(f"- [Missing Values Heatmap]({os.path.basename(correlation_matrix_path)})\n")
        for histogram_path in histogram_paths:
            f.write(f"- [{os.path.basename(histogram_path)}]({os.path.basename(histogram_path)})\n\n")
        f.write("## Insights and Observations\n")
        f.write(insights + "\n")
    print(f"README file created at {readme_path}")

def main():
    dataset_path = input("Enter the path to your CSV dataset: ")
    load_and_analyze_dataset(dataset_path)

if __name__ == "__main__":
    main()
