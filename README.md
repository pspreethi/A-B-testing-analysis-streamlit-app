# A/B Testing Analysis Streamlit App  

A Streamlit-based application for analyzing A/B testing results with statistical tests, machine learning predictions, and automated PowerPoint report generation.

launch App: https://a-b-testing-analysis.streamlit.app/

## Features  

- **Upload Data**: Supports datasets with required columns:  
  - `user_id`: Unique identifier for each user  
  - `group`: A/B test group (`A` = control, `B` = treatment)  
  - `engagement`: Binary (`0/1`) indicating user interaction  
  - `conversion`: Binary (`0/1`) indicating user conversion  
  - `metric`: Any numeric column representing user activity (e.g., `time_spent`)  

- **Select Statistical Tests**: Choose from multiple statistical tests to analyze significance:  
  - Chi-Square Test  
  - Fisher’s Exact Test  
  - Z-Test for Proportions  
  - T-Test  
  - Mann-Whitney U Test  

- **Train Machine Learning Model**: Train an `XGBoost` model to predict conversion outcomes based on user engagement and other features.  

- **Generate PowerPoint Reports**: Get structured reports with statistical insights and ML results, using a customizable `template.pptx`.  

## Repository Structure

- **A-B-testing-analysis-streamlit-app/**
  - `.gitignore` - Files to be ignored in git commits
  - `README.md` - Project documentation
  - `ab_test_sample_data_3000.csv` - Sample dataset for testing
  - `analysis.py` - Main Streamlit app for analysis
  - `requirements.txt` - Dependencies for the project
  - `template.pptx` - PowerPoint template for reports

## Installation Guide

### 1. Clone the repository

git clone https://github.com/pspreethi/A-B-testing-analysis-streamlit-app.git <br>
cd A-B-testing-analysis-streamlit-app

### 2. Create a virtual environment (Optional but Recommended)
On Mac/Linux:  
python -m venv venv<br>
source venv/bin/activate <br>
On Windows:  
python -m venv venv <br>
venv\Scripts\activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Run the Streamlit app

streamlit run analysis.py

## Usage

Upload the dataset in CSV format.  
Map the columns to user_id, group, engagement, conversion, and metric.  
Select a statistical test for significance analysis.  
Train an XGBoost model to predict conversions.  
Generate a PowerPoint report summarizing the results.  

## License
This project is open-source and available under the MIT License.
