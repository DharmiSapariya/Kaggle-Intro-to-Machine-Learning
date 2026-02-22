# Exercise: Explore Your Data - Notes

[![Course](https://img.shields.io/badge/Course-Intro%20to%20ML-blue)](https://www.kaggle.com/learn/intro-to-machine-learning)
[![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF)](https://www.kaggle.com)
[![Exercise](https://img.shields.io/badge/Type-Hands--On%20Exercise-success)](https://www.kaggle.com)

**📚 Course:** Intro to Machine Learning (Kaggle)  
**👨‍🎓 Completed by:** Dharmi Sapariya  
**📖 Topic:** Hands-On Data Exploration Practice  
**📅 Date:** February 22, 2026  
**🎯 Exercise:** Part 2 - Basic Data Exploration

---

## 📑 Table of Contents

- [Overview](#overview)
- [Dataset Information](#dataset-information)
- [Step 1: Loading Data](#step-1-loading-data)
- [Step 2: Exploring the Data](#step-2-exploring-the-data)
- [Step 3: Statistical Analysis](#step-3-statistical-analysis)
- [Key Findings](#key-findings)
- [Code Summary](#code-summary)
- [Insights and Observations](#insights-and-observations)

---

## Overview

This exercise applies the data exploration concepts learned in the tutorial to a real dataset. Instead of the Melbourne housing data used in examples, this exercise uses the **Iowa Housing Dataset** to practice:
- Loading data with Pandas
- Viewing data structure
- Computing summary statistics
- Answering specific questions about the data

**Goal:** Get hands-on experience exploring a dataset before building machine learning models.

---

## Dataset Information

### Iowa Housing Dataset

**File:** `train.csv`  
**Source:** Kaggle - Home Data for ML Course  
**Purpose:** Predict house sale prices based on various features

**Dataset Dimensions:**
- **Rows:** 1,460 houses
- **Columns:** 81 features
- **Target Variable:** SalePrice

**Data Characteristics:**
- Mix of numerical and categorical features
- Contains missing values in several columns
- Real estate data from Iowa
- Houses sold between 2006-2010

---

## Step 1: Loading Data

### Code Implementation

```python
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Load the CSV file into a DataFrame
home_data = pd.read_csv(iowa_file_path)

# Display first 5 rows to verify loading
print(home_data.head())
```

### What This Does

1. **Import Pandas** - Brings in the data manipulation library
2. **Define file path** - Specifies where the data file is located
3. **Read CSV** - Uses `pd.read_csv()` to load the file into a DataFrame
4. **Verify loading** - Prints first 5 rows to confirm data loaded correctly

### Output Preview

```
   Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape
0   1          60       RL         65.0     8450   Pave   NaN      Reg
1   2          20       RL         80.0     9600   Pave   NaN      Reg
2   3          60       RL         68.0    11250   Pave   NaN      IR1
3   4          70       RL         60.0     9550   Pave   NaN      IR1
4   5          60       RL         84.0    14260   Pave   NaN      IR1
```

**Key Observations from head():**
- Data loaded successfully
- Multiple feature types visible (numerical and categorical)
- Some NaN (missing) values present
- Each row represents a house with unique ID

---

## Step 2: Exploring the Data

### Comprehensive Data Exploration

```python
import pandas as pd

iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)

# View first rows
print(home_data.head())

# Get detailed information about columns
print(home_data.info())

# Get statistical summary
print(home_data.describe())
```

### Output: home_data.info()

This command revealed:
- **Total entries:** 1,460 houses
- **Total columns:** 81 features
- **Data types:**
  - `int64` - Integer numerical data
  - `float64` - Decimal numerical data
  - `object` - Text/categorical data

**Missing Values Detected:**
- LotFrontage: 259 missing (1201 non-null out of 1460)
- Alley: 1369 missing (only 91 non-null)
- MasVnrType: 872 missing (588 non-null)
- MasVnrArea: 8 missing (1452 non-null)
- BsmtQual: 37 missing (1423 non-null)
- And several other columns with missing data

**Why This Matters:** Missing values need to be handled before building models, as most algorithms cannot work with missing data.

---

## Step 3: Statistical Analysis

### Summary Statistics Command

```python
# Print summary statistics
print(home_data.describe())
```

### Key Statistics Revealed

**LotArea (Property Size):**
- Count: 1,460 properties
- Mean: 10,516.83 sq ft
- Std Dev: 9,981.26 sq ft (high variation)
- Min: 1,300 sq ft
- Max: 215,245 sq ft (huge outlier!)
- Median (50%): 9,478.5 sq ft

**YearBuilt:**
- Oldest house: 1872
- Newest house: 2010
- Average year: 1971
- Median year: 1973

**SalePrice (Target Variable):**
- Count: 1,460 prices
- Mean: $180,921
- Std Dev: $79,442 (high variation in prices)
- Min: $34,900
- Max: $755,000
- Median: $163,000
- 25th percentile: $129,975
- 75th percentile: $214,000

**OverallQual (Quality Rating 1-10):**
- Mean: 6.1
- Most houses rated between 5-7
- Range: 1 to 10

---

## Key Findings

### Exercise Questions Answered

#### Question 1: Average Lot Size

```python
avg_lot_size = round(home_data["LotArea"].mean())
print(avg_lot_size)
```

**Answer:** 10,517 square feet

**How I got this:**
1. Selected the "LotArea" column
2. Used `.mean()` to calculate average
3. Used `round()` to get nearest integer

#### Question 2: Age of Newest Home

```python
newest_home_age = 2026 - home_data["YearBuilt"].max()
print(newest_home_age)
```

**Answer:** 16 years old

**How I got this:**
1. Found the maximum value in "YearBuilt" column using `.max()`
2. Subtracted from current year (2026)
3. Result: Newest house built in 2010, making it 16 years old

---

## Code Summary

### Complete Working Code

```python
import pandas as pd

# Load the data
iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)

# Explore the data structure
print(home_data.head())        # First 5 rows
print(home_data.info())        # Column information
print(home_data.describe())    # Statistical summary

# Calculate average lot size
avg_lot_size = round(home_data["LotArea"].mean())
print(f"Average Lot Size: {avg_lot_size} sq ft")

# Calculate age of newest home
newest_home_age = 2026 - home_data["YearBuilt"].max()
print(f"Newest Home Age: {newest_home_age} years")
```

### Key Methods Used

| Method | Purpose | Returns |
|--------|---------|---------|
| `pd.read_csv()` | Load CSV file | DataFrame |
| `.head()` | View first rows | DataFrame |
| `.info()` | Column details | None (prints info) |
| `.describe()` | Statistics | DataFrame |
| `.mean()` | Calculate average | Float |
| `.max()` | Find maximum | Value |
| `round()` | Round to integer | Integer |

---

## Insights and Observations

### Data Quality Issues Discovered

1. **Missing Values**
   - Multiple columns have missing data
   - Alley column missing 94% of values
   - Will need imputation strategy before modeling

2. **Outliers Present**
   - LotArea has extreme values (max: 215,245 vs mean: 10,517)
   - SalePrice ranges from $34,900 to $755,000
   - May need outlier handling

3. **Data Distribution**
   - Houses built between 1872-2010 (138-year span)
   - Most houses built around 1970s
   - Prices show right-skewed distribution

### Important Considerations

**About the Dataset Age:**
The newest house is from 2010, making the dataset potentially outdated for current predictions. Two possible explanations:

1. **No new construction** - Area stopped building new houses
2. **Old data** - Dataset collected in 2010, no newer data available

**Impact on Model:**
- If explanation #1: Model might still be valid for that area
- If explanation #2: Model might not capture current market trends
- Need to consider data freshness when making predictions

### What I Learned

✓ **How to load data** - Using `pd.read_csv()` with file paths  
✓ **How to explore data structure** - Using `.head()`, `.info()`, `.describe()`  
✓ **How to calculate statistics** - Using `.mean()`, `.max()`, and other aggregations  
✓ **How to select columns** - Using `dataframe["column_name"]` syntax  
✓ **How to identify data issues** - Spotting missing values and outliers  
✓ **How to answer questions with code** - Translating questions into Pandas operations

### Skills Practiced

1. **Data Loading** - Successfully loaded external CSV file
2. **Data Inspection** - Used multiple methods to understand the data
3. **Statistical Analysis** - Calculated meaningful statistics
4. **Problem Solving** - Answered specific questions using code
5. **Data Quality Assessment** - Identified missing values and outliers

---

## Next Steps

Based on this exploration, the next steps in the ML workflow would be:

1. **Feature Selection** - Choose which of the 81 columns to use
2. **Handle Missing Values** - Decide strategy for NaN values
3. **Build First Model** - Create a simple baseline model
4. **Validate Model** - Check prediction accuracy
5. **Improve Model** - Iterate based on results

---

## 📝 Personal Reflection

**What Went Well:**
- Successfully loaded and explored the Iowa housing dataset
- Correctly calculated required statistics
- Identified important data quality issues
- Gained hands-on experience with Pandas methods

**Key Takeaways:**
- Data exploration is essential before modeling
- Real datasets have missing values and outliers
- Summary statistics reveal important patterns
- Understanding data age/freshness matters for predictions

**What I Would Do Differently:**
- Could explore more columns in detail
- Could visualize distributions of key features
- Could investigate missing value patterns more deeply
- Could compare this to the Melbourne dataset used in tutorials

---

## 🔗 Related Resources

- [Previous Notes: Basic Data Exploration](./basic-data-exploration-notes.md)
- [Kaggle Exercise: Explore Your Data](https://www.kaggle.com/code/dharmisapariya/exercise-explore-your-data)
- [Original Tutorial by Dan Becker](https://www.kaggle.com/code/dansbecker/basic-data-exploration)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

## 📄 License

These are personal learning notes. Original exercise and dataset by Kaggle.

---

**Personal Note:** These notes document my hands-on practice with data exploration on the Iowa Housing dataset. This exercise reinforced the concepts learned in the tutorial and gave me practical experience with real-world data challenges like missing values and outliers. This is a crucial step before building any machine learning model.

**⭐ Part of my Machine Learning learning journey!**
