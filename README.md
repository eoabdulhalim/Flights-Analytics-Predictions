# Flights Analytics & Predictions

## Overview
This project focuses on analyzing and predicting flight delays using machine learning and data analytics. The project is divided into three main components: **Exploratory Data Analysis (EDA) and Model Training**, **Production Model Pipeline**, and **Power BI Integration for Visualization and Prediction Reporting**.

The goal of the project is to classify flight delays effectively and integrate the predictions with a comprehensive business intelligence report in **Power BI**. This repository contains all the required scripts, models, and reports to demonstrate the end-to-end solution.

---

## Components

### 1. **Flights Training Model Notebook (`Flights_Training_Model.ipynb`)**
- **Description**: This notebook covers **EDA**, **data visualization**, and the **development of a machine learning model** for flight delay classification.
- **Key Steps**:
  - Performed **Exploratory Data Analysis (EDA)** to understand patterns in the data.
  - Preprocessed the dataset by handling missing values, outliers, and normalizing data.
  - Engineered features to improve the model's predictive power.
  - Trained a machine learning model (e.g., Linear Regression, Random Forest, etc.) using **scikit-learn**.
  - Evaluated the model on testing data using metrics such as accuracy, precision, recall, and F1-score.
- **Output**: 
  - The trained model is saved as `model.pkl`.
  - A fitted scaler is saved as `scaler.pkl` for data normalization.

---

### 2. **Flights Model Production Notebook (`flights_model_production.py`)**
- **Description**: This notebook contains all necessary data cleaning, transformation, and feature encoding functions. It also implements the **full pipeline function** for generating predictions on new data.
- **Key Features**:
  - **Data Cleaning**: Handles missing values and normalizes numerical features.
  - **Feature Engineering**: Encodes categorical variables and applies scaling.
  - **Pipeline Implementation**:
    - Loads the pre-trained `model.pkl` and `scaler.pkl`.
    - Applies transformations to raw input data.
    - Generates predictions in a formatted **Pandas DataFrame**.
- **Functionality**:
  - `Model_Pipeline(df, df_hist, scaler, model)`: 
    - Inputs: Current flight data (`df`), historical dynamic data (`df_hist`), scaler, and trained model.
    - Output: Predictions in a well-structured dataframe.
- **Output**: Cleaned and transformed data is used to classify delays and predict outcomes.

---

### 3. **Flights Analytics Power BI Report**
- **Description**: A fully interactive **Power BI Report** that combines the raw data and model predictions to provide business insights.
- **Key Features**:
  - **Data Visualizations**:
    - Flight delay trends by airline, airport, and time.
    - Distribution of delay reasons (e.g., weather, system issues).
    - Departure and arrival performance.
  - **Machine Learning Integration**:
    - Uses Python scripts within Power BI to connect with `flights_model_production.py`.
    - Generates predictions dynamically based on user-provided input data.
  - **Interactive Filtering**:
    - Allows users to filter data by airline, origin, destination, and other dimensions to uncover insights.

---

## Key Highlights
- **End-to-End Workflow**: Covers the entire machine learning lifecycle, from **EDA and model training** to **production deployment** and **business reporting**.
- **Integration with Power BI**: The project demonstrates how machine learning can be seamlessly integrated into Power BI for real-time analytics and predictions.
- **Reusable Code**: All preprocessing and prediction functions are modular and can be applied to other datasets.

---

## Repository Structure
```plaintext
├── Data Model.png                    # Data model representation of the dataset tables
├── Datasets download links           # Contains all Datasets used
├── Flights_Training_Model.ipynb      # EDA, model training, and evaluation
├── Flights_Model_Production.py       # Data cleaning, transformation, and Full-pipeline functions
├── Python Script Code                # Power BI python script to load model and new predictions
├── model.pkl                         # Saved trained model
├── scaler.pkl                        # Saved scaler for feature scaling
├── README.md                         # Project documentation
```

---

## How to Use

### 1. **Setup Python Environment**
- Install the required Python libraries:
  ```bash
  pip install pandas numpy scikit-learn joblib
  ```
- Ensure you have Power BI Desktop installed for the interactive report.

### 2. **Run the Model Pipeline**
- Open `flights_model_production.ipynb`.
- Load the input data (current and historical flight datasets).
- Call the `Model_Pipeline` function to preprocess and predict delays.

### 3. **Power BI Integration**
- Open `flights_analytics.pbix` in Power BI Desktop.
- Ensure the Python environment is configured in Power BI:
  - File → Options and Settings → Python Scripting → Select your Python installation.
  - Do the steps here-> https://docs.google.com/document/d/1XjN8_Lx9jQurBBdvdvVRdX-qqpti3bzcauM59vyq55s/edit?tab=t.0
- Refresh the report to generate real-time predictions and insights.

### 4. **Visualize Insights**
- Explore trends and predictions using the interactive dashboard.
- Use slicers to filter by airlines, airports, and other dimensions.

---

## Sample Dashboard
> ![image](https://github.com/user-attachments/assets/b3d66915-f8f8-4403-9dc3-861672c07181)
link to the Report-> https://app.powerbi.com/view?r=eyJrIjoiOWIyZDFmOWMtYmFlMC00Zjc0LWIzY2YtNWY2YzM5NDBhMTgzIiwidCI6ImVhZjYyNGM4LWEwYzQtNDE5NS04N2QyLTQ0M2U1ZDc1MTZjZCIsImMiOjh9
---

## Tools and Technologies
- **Languages**: Python (Pandas, NumPy, Scikit-learn, Matplotlib)
- **Machine Learning**: Random Forest Regressor, Pipelines
- **Business Intelligence**: Power BI Desktop
- **Version Control**: GitHub

---

## Contact
For inquiries or collaboration opportunities, please feel free to reach out:
- **Email**: eoabdulhalim@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/eoabdulhalim
