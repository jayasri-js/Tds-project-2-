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
import argparse
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

# Load dataset and perform data analysis
def load_and_analyze_dataset(dataset_path):
    """
    This function loads a dataset, performs basic analysis like detecting missing values,
    skewness, kurtosis, and generates visualizations.
    """
    encodings = ['utf-8', 'utf-16', 'ISO-8859-1']  # Different encodings to try while loading dataset
    df = None  # Initialize DataFrame

    # Directory to save output files (like plots)
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)

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
    heatmap_path = os.path.join(output_dir, "missing_values_heatmap.png")
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
        histogram_path = os.path.join(output_dir, f"{column}_distribution.png")
        plt.savefig(histogram_path)
        plt.close()
        histogram_paths.append(histogram_path)

    # Correlation matrix for numeric columns
    correlation_matrix = df[numeric_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    correlation_matrix_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(correlation_matrix_path)
    plt.close()

    # Outlier detection using IQR (Interquartile Range) method
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).sum()

    # Send analysis data to LLM for insights
   
    def send_to_llm(df, numeric_columns, missing_values, skewness, kurtosis, correlation_matrix, outliers):
    """
    This function sends data to GPT-4o-mini to generate precise and actionable insights based on the analysis.
    """
    try:
        # Create a structured prompt with improved clarity and specificity
        prompt = f"""
        You are a data analysis assistant. Here's the summary of the dataset analysis:
        
        1. **Overview**:
            - Total rows: {len(df)}
            - Total columns: {len(df.columns)}
            - Numeric columns analyzed: {len(numeric_columns)}
        
        2. **Missing Values**:
        {missing_values.to_string()}
        
        3. **Numeric Feature Analysis**:
            - **Skewness**:
            {skewness.to_string()}
            - **Kurtosis**:
            {kurtosis.to_string()}
        
        4. **Correlation Matrix**:
        ```
        {correlation_matrix.to_string()}
        ```
        
        5. **Outliers Detected**:
        ```
        {outliers.to_string()}
        ```

        Using this analysis, please provide:
        - A concise description of the dataset's structure and quality.
        - Patterns or trends observed in the numeric features.
        - Key insights or relationships derived from the correlation matrix.
        - Suggestions for handling missing values and outliers.
        - Actions or strategies for improving data quality and preparing it for further analysis.
        - Any additional hypotheses or observations based on the data summary.
        """

        # Send the prompt to GPT-4o-mini
        data = {
            "model": "gpt-4o-mini",  # Use the mini version identifier
            "messages": [
                {"role": "system", "content": "You are a highly accurate and systematic data analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,  # Limit tokens to focus on concise outputs
            "temperature": 0.5,  # Lower temperature for consistent and fact-based responses
            "top_p": 1.0,       # Ensure deterministic responses
        }

        headers = {
            "Authorization": f"Bearer {AIPROXY_TOKEN}",  # Ensure the correct token is used
            "Content-Type": "application/json"
        }

        # Make the API request
        response = requests.post(API_URL, json=data, headers=headers)

        # Handle API response
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error sending data to LLM: {e}"


def create_readme(output_dir, missing_values, skewness, kurtosis, correlation_matrix_path, histogram_paths, llm_response):
    """
    This function generates a README file to document the analysis results and LLM insights.
    """
    readme_path = os.path.join(output_dir, "README.md")
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
    Handles both command-line arguments and automated paths.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze a dataset from a CSV file.")
    parser.add_argument(
        "dataset", nargs="?", type=str,  # Allow positional dataset path
        help="Path to the input CSV dataset. If not provided, the DATASET_PATH environment variable will be used."
    )
    parser.add_argument(
        "--dataset", type=str, default=None,  # Allow optional dataset argument
        help="Path to the input CSV dataset. Overrides positional argument or environment variable."
    )
    args = parser.parse_args()

    # Determine dataset path
    dataset_path = args.dataset or os.getenv("DATASET_PATH")
    if not dataset_path:
        raise ValueError(
            "No dataset path provided. Provide a dataset path as a positional argument, "
            "use the --dataset argument, or set the DATASET_PATH environment variable."
        )

    # Call the analysis function with the resolved dataset path
    load_and_analyze_dataset(dataset_path)

if __name__ == "__main__":
    main()

     
