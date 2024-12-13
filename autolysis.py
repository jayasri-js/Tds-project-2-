import os

# Set the AIPROXY_TOKEN in the environment
os.environ["AIPROXY_TOKEN"] = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZHMzMDAwMDY2QGRzLnN0dWR5LmlpdG0uYWMuaW4ifQ.bwU8TOqoWKh_WUduMK1D7ZxQz-WlFCAcrxb6pkxV7ls"

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import numpy as np

# Set up the proxy API URL
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Ensure the AIPROXY_TOKEN is available in your environment variables
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# Function to load and analyze the dataset
def load_and_analyze_dataset(dataset_path):
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path)
        print("Dataset loaded successfully!")

        # Basic summary statistics
        summary = df.describe()

        # Check for missing values
        missing_values = df.isnull().sum()

        # Check for skewness and kurtosis (for numeric columns)
        skewness = df.select_dtypes(include=['number']).skew()
        kurtosis = df.select_dtypes(include=['number']).kurtosis()

        # Visualize missing values as a heatmap
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title("Missing Values Heatmap")
        plt.show()

        # Visualizing numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
            # Create histograms for numeric columns
            for column in numeric_columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[column], kde=True, bins=30)
                plt.title(f'{column} Distribution')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                plt.show()

        # Outlier detection: IQR method
        Q1 = df[numeric_columns].quantile(0.25)
        Q3 = df[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).sum()

        # Create correlation matrix and heatmap
        correlation_matrix = df[numeric_columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

        # Clustering (optional for numeric data)
        from sklearn.cluster import KMeans

        # Dropping rows with missing values before clustering
        df_cleaned = df.dropna(subset=numeric_columns)  # Only drop rows where numeric columns have missing values
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(df_cleaned[numeric_columns])  # Fit clustering only on cleaned data
        df['Cluster'] = np.nan  # Initialize the 'Cluster' column with NaN
        df.loc[df_cleaned.index, 'Cluster'] = kmeans.labels_  # Assign cluster labels to the original dataframe

        # Prepare the prompt for the LLM to generate a story
        send_to_llm(df, numeric_columns, missing_values, skewness, kurtosis, correlation_matrix, outliers)

    except Exception as e:
        print(f"Error loading or analyzing the dataset: {e}")

# Function to send dataset analysis to LLM for insights
def send_to_llm(df, numeric_columns, missing_values, skewness, kurtosis, correlation_matrix, outliers):
    try:
        # Prepare the prompt
        prompt = f"""
        I have analyzed a dataset with the following characteristics:
        
        **Missing Values:** {missing_values}
        **Skewness in Numeric Features:** {skewness}
        **Kurtosis in Numeric Features:** {kurtosis}
        **Correlation Matrix:** {correlation_matrix.to_string()}
        **Outliers (IQR Method):** {outliers}
        
        Based on this data, please answer the following questions:
        1. **What data did you receive? Briefly describe the dataset.**
        2. **What analysis was carried out on the dataset?**
        3. **What insights did you discover from the analysis?**
        4. **What are the implications of your findings, i.e., what should be done with the insights?**
        
        Please generate a detailed and structured response to each of these questions.
        """

        # Set up the payload for the API request
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500
        }

        # Set up the headers for the API request with the AIPROXY_TOKEN
        headers = {
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json"
        }

        # Send the API request
        response = requests.post(API_URL, json=data, headers=headers)

        if response.status_code == 200:
            # Extracting the generated insights from the response
            insights = response.json()['choices'][0]['message']['content'].strip()
            print("\nLLM Generated Insights:\n")
            print(insights)
        else:
            print(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Error sending data to LLM: {e}")

# Main function to run the complete flow
def main():
    dataset_path = input("Enter the path to your CSV dataset: ")
    load_and_analyze_dataset(dataset_path)

if __name__ == "__main__":
    main()
