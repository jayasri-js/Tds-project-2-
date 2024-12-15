# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib",
#   "seaborn",
#   "pandas",
#   "numpy",
#   "requests"
# ]
# ///

import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set non-GUI backend for matplotlib
import matplotlib
matplotlib.use('Agg')

# Read API token from environment variable (required for LLM communication)
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable is not set.")

API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Function to load dataset and perform analysis
def load_and_analyze_dataset(dataset_path):
    """
    Load a dataset, perform analysis, and generate outputs like visualizations and a README file.
    """
    encodings = ['utf-8', 'utf-16', 'ISO-8859-1']
    df = None

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Attempt to load the dataset with various encodings
    for encoding in encodings:
        try:
            df = pd.read_csv(dataset_path, encoding=encoding)
            print(f"Dataset loaded successfully with {encoding} encoding!")
            break
        except UnicodeDecodeError:
            print(f"Failed with {encoding} encoding. Trying the next.")
        except Exception as e:
            print(f"Error loading dataset: {e}")

    if df is None:
        raise ValueError("Failed to load dataset with all attempted encodings.")

    # Perform analysis
    summary = df.describe()
    missing_values = df.isnull().sum()
    skewness = df.select_dtypes(include=['number']).skew()
    kurtosis = df.select_dtypes(include=['number']).kurtosis()

    # Generate visualizations
    generate_visualizations(df, output_dir)

    # Correlation matrix
    numeric_columns = df.select_dtypes(include=['number']).columns
    correlation_matrix = df[numeric_columns].corr()

    # Outlier detection
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).sum()

    # LLM insights
    llm_response = send_to_llm(df, missing_values, skewness, kurtosis, correlation_matrix, outliers)

    # Generate README
    create_readme(output_dir, missing_values, skewness, kurtosis, correlation_matrix, llm_response)

# Generate visualizations
def generate_visualizations(df, output_dir):
    """Generate heatmaps and histograms from the dataset."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    heatmap_path = os.path.join(output_dir, "missing_values_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()

    numeric_columns = df.select_dtypes(include=['number']).columns
    for column in numeric_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True, bins=30)
        plt.title(f"{column} Distribution")
        histogram_path = os.path.join(output_dir, f"{column}_distribution.png")
        plt.savefig(histogram_path)
        plt.close()

# Send analysis data to LLM
def send_to_llm(df, missing_values, skewness, kurtosis, correlation_matrix, outliers):
    """Send dataset analysis results to LLM for insights."""
    prompt = f"""
    Dataset Analysis Results:
    - Missing Values: {missing_values}
    - Skewness: {skewness}
    - Kurtosis: {kurtosis}
    - Correlation Matrix: {correlation_matrix.to_string()}
    - Outliers: {outliers}

    Provide insights based on the above analysis.
    """

    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500
    }

    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, json=data, headers=headers)

    if response.status_code == 200:
        return response.json().get('choices', [{}])[0].get('message', {}).get('content', 'No response from LLM.')
    else:
        return f"Error {response.status_code}: {response.text}"

# Create README file
def create_readme(output_dir, missing_values, skewness, kurtosis, correlation_matrix, llm_response):
    """Generate a README file summarizing the analysis."""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Dataset Analysis Report\n\n")
        f.write("## Missing Values\n")
        f.write(missing_values.to_string() + "\n\n")
        f.write("## Skewness\n")
        f.write(skewness.to_string() + "\n\n")
        f.write("## Kurtosis\n")
        f.write(kurtosis.to_string() + "\n\n")
        f.write("## Correlation Matrix\n")
        f.write(correlation_matrix.to_string() + "\n\n")
        f.write("## LLM Insights\n")
        f.write(llm_response + "\n")

    print(f"README created at {readme_path}")

# Main function
def main():
    dataset_path = input("Enter the path to your dataset (CSV file): ").strip()
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    load_and_analyze_dataset(dataset_path)

if __name__ == "__main__":
    main()


  
      
         
