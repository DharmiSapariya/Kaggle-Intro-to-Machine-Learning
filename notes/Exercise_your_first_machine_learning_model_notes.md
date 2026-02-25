# Exercise: Your First Machine Learning Model - Notes

[![Course](https://img.shields.io/badge/Course-Intro%20to%20ML-blue)](https://www.kaggle.com/learn/intro-to-machine-learning)
[![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF)](https://www.kaggle.com)
[![Exercise](https://img.shields.io/badge/Type-Hands--On%20Exercise-success)](https://www.kaggle.com)

**📚 Course:** Intro to Machine Learning (Kaggle)  
**👨‍🎓 Completed by:** Dharmi Sapariya  
**📖 Topic:** Building Your First ML Model - Hands-On Practice  
**📅 Date:** February 25, 2026  
**🎯 Exercise:** Part 4 - Your First Machine Learning Model Exercise  
**✅ Status:** All Steps Completed Successfully

---

## 📑 Table of Contents

- [Overview](#overview)
- [Dataset Information](#dataset-information)
- [Step 1: Specify Prediction Target](#step-1-specify-prediction-target)
- [Step 2: Create Feature Matrix (X)](#step-2-create-feature-matrix-x)
- [Step 3: Specify and Fit Model](#step-3-specify-and-fit-model)
- [Step 4: Make Predictions](#step-4-make-predictions)
- [Results Analysis](#results-analysis)
- [Complete Code Summary](#complete-code-summary)
- [Key Learnings](#key-learnings)

---

## Overview

This exercise applies the concepts from the "Your First Machine Learning Model" tutorial to the **Iowa Housing Dataset**. The goal is to build a complete machine learning workflow from scratch:

1. Select the prediction target (y)
2. Choose features (X)
3. Build a Decision Tree model
4. Train the model
5. Make predictions

**What Makes This Different:**
- Applying concepts independently (not following along)
- Using different dataset than the tutorial (Iowa vs Melbourne)
- Making my own feature selection decisions
- Testing understanding through practice

---

## Dataset Information

### Iowa Housing Dataset

**File:** `train.csv`  
**Path:** `../input/home-data-for-ml-course/train.csv`

**Dataset Characteristics:**
- **Rows:** 1,460 houses
- **Columns:** 81 features
- **Target:** SalePrice
- **Purpose:** Predict house sale prices

**Key Difference from Tutorial:**
- Tutorial used Melbourne (Australia) housing data
- Exercise uses Iowa (USA) housing data
- Different features available
- Different price ranges

---

## Step 1: Specify Prediction Target

### Task
Select the target variable (what we want to predict) and save it to variable `y`.

### My Initial Exploration

First, I explored both datasets to understand the structure:

```python
import pandas as pd

# Load Melbourne data (from tutorial)
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# Drop rows with missing values
melbourne_data = melbourne_data.dropna(axis=0)

# Print columns to find the target
print(melbourne_data.columns)

# Set target variable (usually "Price")
y = melbourne_data['Price']
print("\nTarget variable preview:\n", y.head())
```

**Output:**
```
Index(['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',
       'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
       'Longtitude', 'Regionname', 'Propertycount'],
      dtype='object')

Target variable preview:
 1    1035000.0
2    1465000.0
4    1600000.0
6    1876000.0
7    1636000.0
Name: Price, dtype: float64
```

### My Solution for Iowa Dataset

```python
# Step 1: Specify the prediction target
y = home_data.SalePrice

# Check your answer
step_1.check()
```

**Result:** ✅ Correct

### What I Did

1. **Identified the target column:** `SalePrice`
2. **Used dot notation:** `home_data.SalePrice`
3. **Saved to variable y:** Following ML convention

### Key Insights

**Column Name Difference:**
- Melbourne dataset: `Price`
- Iowa dataset: `SalePrice`
- Both represent the same concept (house sale price)

**Why Use Dot Notation?**
- Cleaner syntax than bracket notation
- Works when column name has no spaces
- More Pythonic and readable

---

## Step 2: Create Feature Matrix (X)

### Task
Create a DataFrame called `X` with selected predictive features from this list:
- LotArea
- YearBuilt
- 1stFlrSF
- 2ndFlrSF
- FullBath
- BedroomAbvGr
- TotRmsAbvGrd

### My Solution

```python
# Step 2: Create X

# Create the list of features
feature_names = [
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd"
]

# Select data corresponding to features in feature_names
X = home_data[feature_names]

# Check your answer
step_2.check()
```

**Result:** ✅ Correct

### Understanding the Features

**What Each Feature Represents:**

| Feature | Meaning | Why It Matters |
|---------|---------|----------------|
| `LotArea` | Lot size in square feet | Larger lots typically = higher prices |
| `YearBuilt` | Original construction year | Newer homes often worth more |
| `1stFlrSF` | First floor square feet | More space = higher price |
| `2ndFlrSF` | Second floor square feet | Additional living space adds value |
| `FullBath` | Number of full bathrooms | More bathrooms = higher price |
| `BedroomAbvGr` | Bedrooms above ground | More bedrooms = higher price |
| `TotRmsAbvGrd` | Total rooms above ground | Overall size indicator |

### Reviewing the Data

```python
# Review data

# Print summary statistics
print(X.describe())

# Print the first few rows
print(X.head())
```

**Statistical Summary Output:**
```
             LotArea    YearBuilt     1stFlrSF     2ndFlrSF     FullBath
count    1460.000000  1460.000000  1460.000000  1460.000000  1460.000000
mean    10516.828082  1971.267808  1162.626712   346.992466     1.565068
std      9981.264932    30.202904   386.587738   436.528436     0.550916
min      1300.000000  1872.000000   334.000000     0.000000     0.000000
25%      7553.500000  1954.000000   882.000000     0.000000     1.000000
50%      9478.500000  1973.000000  1087.000000     0.000000     2.000000
75%     11601.500000  2000.000000  1391.250000   728.000000     2.000000
max    215245.000000  2010.000000  4692.000000  2065.000000     3.000000

       BedroomAbvGr  TotRmsAbvGrd
count   1460.000000   1460.000000
mean       2.866438      6.517808
std        0.815778      1.625393
min        0.000000      2.000000
25%        2.000000      5.000000
50%        3.000000      6.000000
75%        3.000000      7.000000
max        8.000000     14.000000
```

**First Few Rows Output:**
```
   LotArea  YearBuilt  1stFlrSF  2ndFlrSF  FullBath  BedroomAbvGr  TotRmsAbvGrd
0     8450       2003       856       854         2             3             8
1     9600       1976      1262         0         2             3             6
2    11250       2001       920       866         2             3             6
3     9550       1915       961       756         1             3             7
4    14260       2000      1145      1053         2             4             9
```

### Key Observations

**From Statistics:**
- Average lot size: ~10,517 sq ft
- Houses built between 1872-2010 (average: 1971)
- Many houses don't have a second floor (50% have 0 sq ft)
- Most houses have 2 full bathrooms
- Average of 3 bedrooms per house
- Some houses have 0 bedrooms (likely studios or special cases)

**Data Quality:**
- All 1,460 rows have complete data for these features
- No missing values (count = 1460 for all)
- Good mix of numerical features
- Reasonable value ranges

---

## Step 3: Specify and Fit Model

### Task
Create a DecisionTreeRegressor model and fit it with the data.

### My Solution

```python
# Step 3: Specify and Fit Model

# Import DecisionTreeRegressor from sklearn
from sklearn.tree import DecisionTreeRegressor

# Specify the model
# For reproducibility, set random_state to any number, e.g., 1
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model to the data
iowa_model.fit(X, y)

# Check your answer
step_3.check()
```

**Result:** ✅ Correct

### Understanding Each Part

**Import Statement:**
```python
from sklearn.tree import DecisionTreeRegressor
```
- `sklearn.tree` - Module containing tree-based models
- `DecisionTreeRegressor` - Specific class for regression trees

**Model Creation:**
```python
iowa_model = DecisionTreeRegressor(random_state=1)
```
- Creates an instance of the model
- `random_state=1` - Ensures reproducible results
- Model is defined but not trained yet

**Model Training:**
```python
iowa_model.fit(X, y)
```
- `.fit()` - Trains the model
- `X` - Feature data (7 columns, 1460 rows)
- `y` - Target data (sale prices)
- Model learns patterns from this data

**What Happens During Fit:**
1. Algorithm analyzes relationships between features and prices
2. Creates decision rules (e.g., "if YearBuilt > 2000 and LotArea > 10000...")
3. Builds a tree structure of decisions
4. Stores the learned pattern in `iowa_model`

---

## Step 4: Make Predictions

### Task
Use the trained model to make predictions and save them to a variable called `predictions`.

### My Solution

```python
# Step 4: Make Predictions

# Use the trained model to make predictions
predictions = iowa_model.predict(X)

# Print the predictions
print(predictions)

# Check your answer
step_4.check()
```

**Result:** ✅ Correct

**Output:**
```
[208500. 181500. 223500. ... 266500. 142125. 147500.]
```

### What This Shows

**Prediction Format:**
- Array of predicted prices
- One prediction for each house (1,460 total)
- Values are in dollars
- First few: $208,500, $181,500, $223,500...

**How It Works:**
1. Model takes feature values from X
2. Applies learned decision rules
3. Follows tree branches to reach a prediction
4. Returns predicted price for each house

---

## Results Analysis

### Comparing Predictions to Actual Values

I created a comparison to see how well the model performed:

```python
# Compare the first few predictions to the actual prices
import pandas as pd

# Create a DataFrame to make it easy to compare
comparison = pd.DataFrame({
    "Predicted": predictions,
    "Actual": y
})

# Show the first few rows
print(comparison.head())
```

**Output:**
```
   Predicted  Actual
0   208500.0  208500
1   181500.0  181500
2   223500.0  223500
3   140000.0  140000
4   250000.0  250000
```

### Surprising Discovery

**Perfect Match!** 🎯

The predictions exactly match the actual prices for these houses. This is surprising because:

**Why This Happened:**
- We're predicting on the **training data**
- The model has already "seen" these houses during training
- Decision trees can memorize training data perfectly
- This doesn't mean the model will work well on NEW houses

**Important Lesson:**
- Perfect training accuracy doesn't mean good model
- Need to test on data the model hasn't seen
- This is why we need model validation (next lesson!)
- Real-world evaluation requires separate test data

---

## Complete Code Summary

### Full Working Code

```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Step 1: Load the data
iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)

# Step 2: Specify the prediction target
y = home_data.SalePrice

# Step 3: Create the list of features
feature_names = [
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd"
]

# Step 4: Select data corresponding to features
X = home_data[feature_names]

# Step 5: Review the data
print(X.describe())
print(X.head())

# Step 6: Specify and fit the model
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(X, y)

# Step 7: Make predictions
predictions = iowa_model.predict(X)

# Step 8: Compare predictions to actual values
comparison = pd.DataFrame({
    "Predicted": predictions,
    "Actual": y
})
print(comparison.head())
```

### The Complete Workflow

```
1. Load Data (Iowa housing CSV)
        ↓
2. Select Target (y = SalePrice)
        ↓
3. Choose Features (7 selected features)
        ↓
4. Create Feature Matrix (X)
        ↓
5. Review Data (describe, head)
        ↓
6. Define Model (DecisionTreeRegressor)
        ↓
7. Train Model (fit on X and y)
        ↓
8. Make Predictions (predict on X)
        ↓
9. Compare Results (actual vs predicted)
```

---

## Key Learnings

### Technical Skills Acquired

✅ **Feature Selection**
- Chose 7 relevant features from 81 available
- Understood what each feature represents
- Selected features that logically affect price

✅ **Data Structure Understanding**
- y is a Series (1D - single column)
- X is a DataFrame (2D - multiple columns)
- Both must align (same number of rows)

✅ **Model Building Process**
- Import → Define → Fit → Predict
- Each step has a specific purpose
- Order matters (can't predict before fitting)

✅ **Code Syntax Mastery**
- Dot notation: `home_data.SalePrice`
- Bracket notation: `home_data[feature_names]`
- Method chaining: `model.fit().predict()` (possible but not done here)

### Conceptual Understanding

**What I Now Understand:**

1. **Target vs Features**
   - Target (y) = What we predict
   - Features (X) = What we use to predict
   - Must be separated before modeling

2. **Training vs Predicting**
   - Training = Learning patterns from data
   - Predicting = Applying learned patterns
   - Training happens once, predicting many times

3. **Model Behavior**
   - Decision trees can memorize training data
   - Perfect training accuracy ≠ good model
   - Need separate test data for true evaluation

4. **Scikit-learn Patterns**
   - Consistent API across models
   - All models have `.fit()` and `.predict()`
   - Easy to swap different model types

### Critical Insights

**⚠️ Important Realizations:**

1. **Perfect Predictions are Suspicious**
   - Got 100% accuracy on training data
   - This is actually a warning sign (overfitting)
   - Real evaluation needs unseen data

2. **Feature Selection Matters**
   - Could have used all 81 features
   - Starting with 7 keeps it simple
   - More features ≠ better model

3. **Data Exploration is Essential**
   - Always review data with `.describe()` and `.head()`
   - Statistics revealed data characteristics
   - Found no missing values in chosen features

4. **Convention Helps Communication**
   - Using X and y is universal in ML
   - Makes code readable to others
   - Following standards is good practice

---

## Challenges and Solutions

### Challenge 1: Understanding Feature Selection

**Problem:** Not sure which features to choose from 81 available

**Solution:**
- Started with provided list of 7 features
- Understood what each feature represents
- Used intuition about what affects house prices
- Will learn statistical methods later

### Challenge 2: Bracket vs Dot Notation

**Problem:** When to use `data.column` vs `data['column']`?

**Solution:**
- Dot notation works when column name has no spaces/special characters
- Bracket notation required for names like '1stFlrSF' (starts with number)
- Bracket notation needed for multiple columns: `data[['col1', 'col2']]`

### Challenge 3: Interpreting Perfect Results

**Problem:** Predictions exactly matched actual values - too good to be true?

**Solution:**
- Realized we're predicting on training data
- Learned this isn't true model evaluation
- Understood need for validation (next lesson)
- Recognized overfitting risk

---

## Mistakes Made and Lessons Learned

### Initial Exploration

**What I Did:**
Explored Melbourne data first before working with Iowa data

**Why:**
Wanted to understand the pattern from the tutorial

**Learning:**
Good practice to explore multiple examples before applying independently

### Code Organization

**Improvement Made:**
Added comments explaining each step clearly

**Why It Matters:**
Makes code readable and shows understanding

---

## Performance Metrics (To Be Properly Evaluated)

**Current Observations:**
- Predictions: [208500, 181500, 223500, ...]
- Actual: [208500, 181500, 223500, ...]
- Perfect match on training data

**What This Actually Means:**
- ✅ Model can learn from data
- ✅ Code is working correctly
- ❌ NOT a measure of true performance
- ⚠️ Potential overfitting

**Next Steps:**
- Learn proper validation techniques
- Test on unseen data
- Calculate real accuracy metrics
- Understand overfitting vs underfitting

---

## Comparison: Tutorial vs Exercise

| Aspect | Tutorial (Melbourne) | Exercise (Iowa) |
|--------|---------------------|-----------------|
| **Dataset** | Melbourne housing | Iowa housing |
| **Target Column** | Price | SalePrice |
| **Features Used** | 5 features | 7 features |
| **Missing Values** | Yes (used dropna) | No (in selected features) |
| **Learning Mode** | Follow along | Independent practice |

---

## What's Next

### Immediate Next Steps

1. **Model Validation** (Next Lesson)
   - Learn to evaluate model properly
   - Split data into train/test sets
   - Calculate accuracy metrics
   - Understand overfitting

2. **Improve the Model**
   - Try different features
   - Adjust model parameters
   - Compare different models
   - Optimize performance

3. **Real-World Application**
   - Make predictions on truly new houses
   - Deploy the model
   - Monitor performance
   - Update as needed

---

## Personal Reflection

### What Went Well

✅ Successfully completed all 4 steps on first try  
✅ Code ran without errors  
✅ Understood the workflow clearly  
✅ Made connections between tutorial and exercise  
✅ Explored data before building model  
✅ Added comparison to analyze results  

### Areas for Growth

📈 Need to understand overfitting better  
📈 Want to learn feature engineering  
📈 Should explore more features  
📈 Need to learn proper validation  
📈 Want to try different model types  

### Key Takeaway

Building a machine learning model is surprisingly straightforward with the right tools and workflow. The hard parts aren't writing the code - they're understanding the data, choosing good features, and properly evaluating performance. The perfect training predictions taught me an important lesson: success on training data doesn't guarantee success on new data.

---

## 🔗 Related Resources

- [Previous Notes: Your First ML Model Tutorial](./your-first-machine-learning-model-notes.md)
- [Exercise Notebook on Kaggle](https://www.kaggle.com/code/dharmisapariya/exercise-your-first-machine-learning-model)
- [scikit-learn DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [Next: Model Validation](https://www.kaggle.com/dansbecker/model-validation)

---

## 📄 License

These are personal learning notes. Original exercise by Kaggle.

---

**✅ Exercise Completed Successfully - Ready for Model Validation!**

**💡 Main Insight:** Perfect predictions on training data revealed the importance of proper validation - a critical lesson that sets up the next topic perfectly.
