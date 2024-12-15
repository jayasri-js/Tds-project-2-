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
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend



import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read API token from environment variable
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable is not set.")

API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

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

    # Histograms
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

    # Send to LLM and get insights
    llm_response = send_to_llm(df, numeric_columns, missing_values, skewness, kurtosis, correlation_matrix, outliers)

    # Create README file
    create_readme(output_dir, missing_values, skewness, kurtosis, correlation_matrix_path, histogram_paths, llm_response)

def send_to_llm(df, numeric_columns, missing_values, skewness, kurtosis, correlation_matrix, outliers):
    try:
        prompt = f"""
        I have analyzed a dataset with the following characteristics:
        
        **Missing Values:** {missing_values}
        **Skewness in Numeric Features:** {skewness}
        **Kurtosis in Numeric Features:** {kurtosis}
        **Correlation Matrix:** {correlation_matrix.to_string()}
        **Outliers (IQR Method):** {outliers}
        
        Please answer the following:
        1. **The data you received:** Briefly describe the dataset.
        2. **The analysis you carried out:** Summarize the key steps.
        3. **The insights you discovered:** Highlight the main findings.
        4. **The implications of your findings:** What actions should be taken based on the insights?
        """

        data = {
            "model": "gpt-4o-mini",
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
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error sending data to LLM: {e}"

def create_readme(output_dir, missing_values, skewness, kurtosis, correlation_matrix_path, histogram_paths, llm_response):
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
        f.write("## LLM-Generated Insights\n")
        f.write(llm_response + "\n")
    print(f"README file created at {readme_path}")

def main():
    # No user input for dataset path, it's predefined to be used directly
    dataset_path = "your_dataset.csv"  # Change this to the actual dataset file path
    load_and_analyze_dataset(dataset_path)

if __name__ == "__main__":
    main()
