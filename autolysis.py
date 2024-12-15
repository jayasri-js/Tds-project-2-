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

# Set non-GUI backend for matplotlib (useful for running scripts in non-interactive environments)
import matplotlib
matplotlib.use('Agg')

# Read API token from environment variable (required for LLM communication)
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN environment variable is not set.")

API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Directory to save output files (like plots and README)
OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_analyze_dataset():
    """
    This function dynamically prompts the user for a CSV file path, loads the dataset, 
    performs basic analysis, and generates visualizations and a README report.
    """
    # Prompt user for dataset path
    dataset_path = input("Enter the path to your CSV file: ").strip()

    if not os.path.exists(dataset_path):
        print(f"Error: File '{dataset_path}' does not exist.")
        return

    encodings = ['utf-8', 'utf-16', 'ISO-8859-1']  # Different encodings to try while loading dataset
    df = None  # Initialize DataFrame

    # Try loading the dataset with different encodings until it succeeds
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
        return  # Exit if dataset is not loaded

    # Perform basic data analysis
    summary = df.describe()  # Get summary statistics for numerical columns
    missing_values = df.isnull().sum()  # Count missing values in each column
    skewness = df.select_dtypes(include=['number']).skew()  # Skewness of numerical columns
    kurtosis = df.select_dtypes(include=['number']).kurtosis()  # Kurtosis of numerical columns

    # Visualize missing values using a heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    heatmap_path = os.path.join(OUTPUT_DIR, "missing_values_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()

    # Create histograms for numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    histogram_paths = []
    for column in numeric_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True, bins=30)
        plt.title(f'{column} Distribution')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        histogram_path = os.path.join(OUTPUT_DIR, f"{column}_distribution.png")
        plt.savefig(histogram_path)
        plt.close()
        histogram_paths.append(histogram_path)

    # Correlation matrix for numeric columns
    correlation_matrix = df[numeric_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    correlation_matrix_path = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
    plt.savefig(correlation_matrix_path)
    plt.close()

    # Outlier detection using IQR (Interquartile Range) method
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).sum()

    # Send analysis data to LLM for insights
    llm_response = send_to_llm(df, numeric_columns, missing_values, skewness, kurtosis, correlation_matrix, outliers)

    # Create a README file with analysis summary and LLM response
    create_readme(missing_values, skewness, kurtosis, correlation_matrix_path, histogram_paths, llm_response)

def send_to_llm(df, numeric_columns, missing_values, skewness, kurtosis, correlation_matrix, outliers):
    """
    This function sends data to a language model (LLM) to generate insights based on the analysis.
    """
    try:
        # Create a structured prompt to send to the LLM
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

        # Send the prompt to the API
        data = {
            "model": "gpt-4",  # Model to use (make sure it's valid)
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

        # Make API request
        response = requests.post(API_URL, json=data, headers=headers)

        # Check for successful response
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error sending data to LLM: {e}"

def create_readme(missing_values, skewness, kurtosis, correlation_matrix_path, histogram_paths, llm_response):
    """
    This function generates a README file to document the analysis results and LLM insights.
    """
    readme_path = os.path.join(OUTPUT_DIR, "README.md")
    with open(readme_path, "w") as f:
        # Write dataset analysis overview to the README
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
    """
    Main function to execute the dataset analysis process.
    """
    load_and_analyze_dataset()

if __name__ == "__main__":
    main()
