## Data preparation and visualization -  Global Mobile Phone Addiction

## Introductory Information

#### University of Prishtina - Faculty of Computer and Software Engineering  
#### Master’s Program in Computer and Software Engineering  
#### Professor: Dr.Sc. Mërgim H. HOTI

---

**Group:** 10  

**Team Members:**  
- Erza Bërbatovci  
- Leotrim Halimi  
- Rinor Ukshini  

---

### Dataset:  
[Global Mobile Phone Addiction Dataset](https://www.kaggle.com/datasets/khushikyad001/global-mobile-phone-addiction-dataset)  
Source: Kaggle — dataset containing mobile phone addiction survey data with demographic and behavioral information.  

---

## Project Setup

Follow these steps to run the project and the Jupyter Notebook.

```bash

Reminder: Add Jupyter Extension on VS Code

#  Create directory
mkdir PVDH-G10
cd PVDH-G10

# Clone the repository
git clone https://github.com/Leotrimii1/PVDH-G10.git


# Create a virtual environment
python3 -m venv ./venv

# Activate the virtual environment
# macOS/Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\activate

# Install project requirements
pip install -r requirements.txt

# Open your .ipynb notebook file. When you attempt to run a cell,
# VS Code will ask you to select a Python environment.
# Select the interpreter that points to your virtual environment:
# ./venv/bin/python


This project focuses on **data pre-processing for preparing data for analysis**.  
It covers the main stages of cleaning, transforming, and selecting data features before building machine learning models.

---


The following steps were completed in this project:

 **Data Collection**  
   - Imported the dataset using `pandas`.

 **Data Type Definition and Quality Check**  
   - Analyzed data types for each column.  
   - Identified missing or incorrect values.

 **Data Integration, Aggregation, and Sampling**  
   - Combined data sources if needed.  
   - Selected data samples for testing and analysis.

 **Data Cleaning and Handling Missing Values**  
   - Removed or filled empty (`NaN`) values.  
   - Applied strategies to maintain dataset integrity.

 **Dimensionality Reduction and Feature Subset Selection**  
   - Used methods like `SelectKBest`, `RFE`, and `RandomForestClassifier` to choose the most important features.

 **Feature Creation (Feature Engineering)**  
   - Created and transformed variables to improve data quality and analysis.

 **Discretization and Binarization**  
   - Converted numeric values into categories (intervals).  
   - Applied binary transformations (True/False) where needed.

 **Data Transformation**  
   - Normalized and standardized values using `MinMaxScaler` and `StandardScaler`.

---
## Technologies Used

- Python 3.12  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

---


