# Basic Data Exploration - Notes

**Course:** Intro to Machine Learning (Kaggle - Dan Becker)  
**Learned by:** Dharmi Sapariya  
**Topic:** Exploring and Understanding Your Data  
**Date:** February 2026

---

## Overview

This tutorial covers the essential first step in any machine learning project: understanding your data. Before building models, you need to explore the dataset to know what you're working with, identify patterns, and catch potential issues.

---

## Why Data Exploration Matters

### Critical First Step

Before building any model, you must:
- **Understand the data structure** - What columns and rows represent
- **Identify data types** - Numerical vs. categorical features
- **Check data quality** - Missing values, outliers, errors
- **Discover patterns** - Relationships between variables
- **Define the problem** - What are you trying to predict?

### Common Issues Found During Exploration

- Missing or null values
- Incorrect data types
- Outliers or unusual values
- Duplicate records
- Inconsistent formatting
- Irrelevant features

---

## Loading Data with Pandas

### Basic Setup

```python
import pandas as pd

# Load data from CSV file
file_path = 'path/to/data.csv'
data = pd.read_csv(file_path)
```

### What is Pandas?

**Pandas** is the primary Python library for data manipulation and analysis. It provides:
- **DataFrames** - 2D table structure (like Excel spreadsheet)
- **Series** - 1D array (single column)
- **Powerful functions** - For cleaning, transforming, and analyzing data

---

## Essential Exploration Commands

### 1. View the Data

```python
# First few rows
data.head()  # Default: first 5 rows
data.head(10)  # First 10 rows

# Last few rows
data.tail()  # Default: last 5 rows
```

**Purpose:** Get a quick sense of what the data looks like.

### 2. Dataset Shape

```python
# Get dimensions
data.shape  # Returns (rows, columns)

# Example output: (1460, 81)
# Means: 1460 rows, 81 columns
```

**Purpose:** Understand the size of your dataset.

### 3. Column Information

```python
# List all column names
data.columns

# Get detailed info about each column
data.info()
```

**Purpose:** See what features are available and their data types.

### 4. Data Types

```python
# Check data types of all columns
data.dtypes
```

**Common Data Types:**
- `int64` - Integer numbers
- `float64` - Decimal numbers
- `object` - Text/strings (often categorical)
- `bool` - True/False values
- `datetime64` - Dates and times

### 5. Summary Statistics

```python
# Statistical summary of numerical columns
data.describe()
```

**What describe() Shows:**
- `count` - Number of non-null values
- `mean` - Average value
- `std` - Standard deviation (spread)
- `min` - Minimum value
- `25%` - First quartile
- `50%` - Median (middle value)
- `75%` - Third quartile
- `max` - Maximum value

**Purpose:** Quickly spot outliers and understand value ranges.

---

## Working with Specific Columns

### Selecting Columns

```python
# Single column (returns Series)
data['ColumnName']
data.ColumnName  # Alternative syntax

# Multiple columns (returns DataFrame)
data[['Column1', 'Column2', 'Column3']]
```

### Example: Housing Data

```python
# Select price column
prices = data['SalePrice']

# Select multiple features
features = data[['LotArea', 'YearBuilt', 'BedroomAbvGr']]
```

---

## Understanding Your Target Variable

### What is a Target?

The **target** (or **label**) is the variable you want to predict.

**Examples:**
- Predicting house prices → Target: `SalePrice`
- Predicting loan default → Target: `DefaultStatus`
- Classifying images → Target: `Category`

### Exploring the Target

```python
# For the Melbourne Housing dataset
home_prices = data['Price']

# View statistics
print(home_prices.describe())

# View first few values
print(home_prices.head())

# Check for missing values
print(home_prices.isnull().sum())
```

---

## Handling Missing Data

### Detecting Missing Values

```python
# Count missing values per column
data.isnull().sum()

# Percentage of missing values
(data.isnull().sum() / len(data)) * 100

# Visualize missing data
import missingno as msno
msno.matrix(data)
```

### Why Missing Data Matters

- Models can't handle missing values directly
- Missing patterns might be informative
- Different strategies for different situations

### Common Strategies (Covered in Later Lessons)

1. **Drop rows** with missing values
2. **Drop columns** with too many missing values
3. **Imputation** - Fill with mean, median, or mode
4. **Advanced imputation** - Predict missing values

---

## Key Pandas Methods Summary

| Method | Purpose | Returns |
|--------|---------|---------|
| `head(n)` | View first n rows | DataFrame |
| `tail(n)` | View last n rows | DataFrame |
| `shape` | Get dimensions | Tuple (rows, cols) |
| `columns` | List column names | Index object |
| `dtypes` | Show data types | Series |
| `info()` | Detailed column info | None (prints) |
| `describe()` | Statistical summary | DataFrame |
| `isnull()` | Check for missing values | DataFrame of bools |
| `sum()` | Sum values (works with bools) | Series/scalar |

---

## Practical Example: Melbourne Housing Data

### Step-by-Step Exploration

```python
# 1. Load the data
import pandas as pd
melbourne_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# 2. Check the shape
print(melbourne_data.shape)
# Output: (13580, 21) - 13,580 houses, 21 features

# 3. View first rows
print(melbourne_data.head())

# 4. See column names
print(melbourne_data.columns)

# 5. Get statistical summary
print(melbourne_data.describe())

# 6. Check data types
print(melbourne_data.dtypes)

# 7. Look at the target variable
print(melbourne_data['Price'].describe())
```

### What You Learn

- Dataset has 13,580 properties
- 21 different features about each property
- Price is the target variable
- Some columns have missing values
- Mix of numerical and categorical features

---

## Common Data Types in ML Datasets

### Numerical Features

**Continuous:**
- Price: $150,000, $275,500
- Area: 1,234.5 sq ft
- Temperature: 72.3°F

**Discrete:**
- Number of rooms: 3, 4, 5
- Year built: 1998, 2005
- Count of items: 10, 25, 100

### Categorical Features

**Nominal (no order):**
- Color: Red, Blue, Green
- Type: House, Apartment, Condo
- Location: North, South, East, West

**Ordinal (has order):**
- Size: Small, Medium, Large
- Grade: A, B, C, D, F
- Condition: Poor, Fair, Good, Excellent

---

## Best Practices for Data Exploration

### ✓ Do's

1. **Always start with head()** - See actual data examples
2. **Check data types** - Ensure they make sense
3. **Look for missing values** - Plan how to handle them
4. **Examine summary statistics** - Spot obvious issues
5. **Understand the target** - Know what you're predicting
6. **Document findings** - Note issues and decisions

### ✗ Don'ts

1. **Don't skip exploration** - Always look before building models
2. **Don't ignore missing data** - Will cause errors later
3. **Don't assume data is clean** - Real data is messy
4. **Don't forget domain knowledge** - Understand what features mean
5. **Don't overlook data types** - Wrong types cause problems

---

## Building Intuition

### Questions to Ask During Exploration

**About the Dataset:**
- How many examples do I have?
- How many features are available?
- Is this enough data for my problem?

**About Features:**
- What does each column represent?
- Which features are numerical vs. categorical?
- Which features seem most relevant?
- Are there any suspicious values?

**About the Target:**
- What am I trying to predict?
- What's the range of values?
- Is the target balanced (for classification)?
- Are there any missing target values?

**About Data Quality:**
- How much data is missing?
- Are there any obvious errors?
- Do the statistics make sense?
- Are there extreme outliers?

---

## Next Steps After Exploration

Once you understand your data:

1. **Select features** - Choose which columns to use
2. **Handle missing values** - Clean the data
3. **Build a model** - Start with a simple baseline
4. **Validate predictions** - Check model performance
5. **Iterate** - Improve based on results

---

## Common Pitfalls to Avoid

❌ **Using data without looking at it** - Always explore first  
❌ **Ignoring data types** - `object` types need special handling  
❌ **Not checking for missing values** - Will cause model errors  
❌ **Assuming all data is useful** - Some columns may be irrelevant  
❌ **Forgetting to check target variable** - Must be present and valid

---

## Key Takeaways

✓ **Data exploration is mandatory** - Never skip this step  
✓ **Pandas is your primary tool** - Learn it well  
✓ **Use describe() and info()** - Quick overview of the dataset  
✓ **Check for missing values** - Handle them before modeling  
✓ **Understand your target** - Know what you're predicting  
✓ **Look at actual data** - Statistics don't show everything  

---

## Code Reference

### Complete Exploration Workflow

```python
import pandas as pd

# Load data
data = pd.read_csv('your_data.csv')

# Basic exploration
print(data.shape)              # Dimensions
print(data.head())             # First rows
print(data.columns)            # Column names
print(data.dtypes)             # Data types
print(data.describe())         # Statistics
print(data.isnull().sum())     # Missing values

# Target variable
print(data['TargetColumn'].describe())
print(data['TargetColumn'].head())

# Specific columns
features = data[['Feature1', 'Feature2', 'Feature3']]
print(features.head())
```

---

**Personal Note:** These notes were created by Dharmi Sapariya while learning from the Kaggle "Intro to Machine Learning" course by Dan Becker. Data exploration is the foundation of every successful ML project. Understanding your data before modeling will save countless hours of debugging and lead to better results.
