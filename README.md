# InsightML

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Scenarios Handled](#scenarios-handled)
- [Contributions](#contributions)

## Project Description

The Machine Learning App is a user-friendly tool for building and evaluating machine learning models on your dataset. By simply uploading a CSV file, you can preprocess the data, select columns for modeling, choose a target column, and apply various machine learning algorithms for both classification and regression tasks. The app handles missing values, encodes categorical variables, splits data into training and test sets, and provides model evaluation metrics.

## Features

- **Data Upload**: Upload CSV files to load your dataset.
- **Missing Value Handling**: Choose to ignore or fill missing values with NaN.
- **Column Selection**: Select columns for modeling, with the option to choose all columns.
- **Target Column Selection**: Define the target column for prediction.
- **Model Selection**: Choose from various machine learning algorithms for classification and regression tasks.
- **Parameter Configuration**: Use default parameters or manually configure model parameters.
- **Train-Test Split**: Split data into training and test sets with custom ratios.
- **Model Training**: Train models using the chosen algorithm and parameters.
- **Model Evaluation**: Evaluate models using accuracy, RMSE, confusion matrix, and cross-validation scores.
- **Upload Test Data**: Upload additional test data to evaluate the model's performance.

## Requirements

- Python 3.7 or higher
- Streamlit
- Pandas
- Scikit-learn
- NumPy

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/meggitt/InsightML.git
   cd machine-learning-app
   ```
2. Create a virtual environment and activate it:
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Configuration

No additional configuration is required. The application is ready to use after installation.

## Running the Application

1. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```
2. Open your web browser and navigate to `http://localhost:8501` to interact with the application.


## Scenarios Handled

- **Missing Data**: The app allows users to either ignore or fill missing values.
- **Categorical Variables**: Automatic encoding of categorical variables.
- **Classification and Regression**: Supports multiple algorithms for both classification and regression tasks.
- **Custom Parameters**: Flexibility to use default or custom parameters for model training.
- **Cross-Validation**: Provides cross-validation scores to evaluate model stability.

## Contributions

Contributions are welcome! Please create an issue or submit a pull request with your changes.
