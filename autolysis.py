import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set your API key directly in the script
AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZHMzMDAwMDY2QGRzLnN0dWR5LmlpdG0uYWMuaW4ifQ.bwU8TOqoWKh_WUduMK1D7ZxQz-WlFCAcrxb6pkxV7ls"

# Define the correct API URL
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def load_dataset(dataset_path):
    """Attempt to load the dataset with multiple encodings."""
    encodings = ['utf-8', 'utf-16', 'ISO-8859-1']
    for encoding in encodings:
        try:
            df = pd.read_csv(dataset_path, encoding=encoding)
            print(f"Dataset loaded successfully with {encoding} encoding!")
            return df
        except UnicodeDecodeError:
            print(f"Error with {encoding} encoding. Trying next encoding...")
        except Exception as e:
            print(f"Error loading dataset with {encoding} encoding: {e}")
    print("Failed to load the dataset with all attempted encodings.")
    return None

def generate_analysis(df, output_dir):
    """Perform analysis and generate visualizations."""
    summary = df.describe()
    missing_values = df.isnull().sum()
    skewness = df.select_dtypes(include=['number']).skew()
    kurtosis = df.select_dtypes(include=['number']).kurtosis()

    # Heatmap of missing values
    heatmap_path = os.path.join(output_dir, "missing_values_heatmap.png")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.savefig(heatmap_path)
    plt.close()

    # Histograms
    numeric_columns = df.select_dtypes(include=['number']).columns
    histogram_paths = []
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
    correlation_matrix_path = os.path.join(output_dir, "correlation_matrix.png")
    if len(numeric_columns) > 1:
        correlation_matrix = df[numeric_columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.savefig(correlation_matrix_path)
        plt.close()
    else:
        correlation_matrix = None
        correlation_matrix_path = None

    # Outlier detection
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).sum()

    return {
        "summary": summary,
        "missing_values": missing_values,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "correlation_matrix": correlation_matrix,
        "correlation_matrix_path": correlation_matrix_path,
        "histogram_paths": histogram_paths,
        "heatmap_path": heatmap_path,
        "outliers": outliers
    }

def send_to_llm(analysis_data):
    """Send analysis data to the LLM for insights."""
    try:
        prompt = f"""
        Dataset Analysis Report:
        - Missing Values:\n{analysis_data['missing_values']}
        - Skewness:\n{analysis_data['skewness']}
        - Kurtosis:\n{analysis_data['kurtosis']}
        - Outliers (IQR Method):\n{analysis_data['outliers']}
        {"- Correlation Matrix:\n" + analysis_data['correlation_matrix'].to_string() if analysis_data['correlation_matrix'] is not None else ""}
        
        Please summarize the findings and suggest actionable insights.
        """

        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a data analyst."},
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
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error sending data to LLM: {e}"

def create_readme(output_dir, analysis_data, llm_response):
    """Generate a README file summarizing the analysis."""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Dataset Analysis Report\n\n")
        f.write("## Summary Statistics\n")
        f.write(analysis_data['summary'].to_string() + "\n\n")
        f.write("## Missing Values\n")
        f.write(analysis_data['missing_values'].to_string() + "\n\n")
        f.write("## Skewness\n")
        f.write(analysis_data['skewness'].to_string() + "\n\n")
        f.write("## Kurtosis\n")
        f.write(analysis_data['kurtosis'].to_string() + "\n\n")
        if analysis_data['correlation_matrix_path']:
            f.write(f"## [Correlation Matrix]({os.path.basename(analysis_data['correlation_matrix_path'])})\n\n")
        f.write("## Visualizations\n")
        f.write(f"- [Missing Values Heatmap]({os.path.basename(analysis_data['heatmap_path'])})\n")
        for histogram_path in analysis_data['histogram_paths']:
            f.write(f"- [{os.path.basename(histogram_path)}]({os.path.basename(histogram_path)})\n")
        f.write("\n## LLM-Generated Insights\n")
        f.write(llm_response)
    print(f"README file created at {readme_path}")

def main():
    dataset_path = input("Enter the path to your CSV dataset: ")
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)

    df = load_dataset(dataset_path)
    if df is not None:
        analysis_data = generate_analysis(df, output_dir)
        llm_response = send_to_llm(analysis_data)
        create_readme(output_dir, analysis_data, llm_response)

if __name__ == "__main__":
    main()
