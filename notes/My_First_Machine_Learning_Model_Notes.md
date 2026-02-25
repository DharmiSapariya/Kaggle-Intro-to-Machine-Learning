# Your First Machine Learning Model - Notes

[![Course](https://img.shields.io/badge/Course-Intro%20to%20ML-blue)](https://www.kaggle.com/learn/intro-to-machine-learning)
[![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF)](https://www.kaggle.com)
[![Author](https://img.shields.io/badge/Instructor-Dan%20Becker-orange)](https://www.kaggle.com/dansbecker)

**📚 Course:** Intro to Machine Learning (Kaggle)  
**👨‍🎓 Learned by:** Dharmi Sapariya  
**📖 Topic:** Building Your First ML Model with Decision Trees  
**📅 Date:** February 2026  
**🎯 Tutorial:** Part 4 - Your First Machine Learning Model

---

## 📑 Table of Contents

- [Overview](#overview)
- [Selecting Data for Modeling](#selecting-data-for-modeling)
- [Choosing the Prediction Target](#choosing-the-prediction-target)
- [Selecting Features](#selecting-features)
- [Building Your Model](#building-your-model)
- [Making Predictions](#making-predictions)
- [Complete Code Workflow](#complete-code-workflow)
- [Key Concepts](#key-concepts)
- [Important Terminology](#important-terminology)

---

## Overview

This tutorial teaches you how to build your first actual machine learning model. You'll learn to:
- Select relevant columns (features) from your dataset
- Define what you're trying to predict (target)
- Build a Decision Tree model using scikit-learn
- Make predictions with your trained model

**Dataset Used:** Melbourne Housing Snapshot (example)  
**Model Type:** Decision Tree Regressor  
**Goal:** Predict house prices based on property features

---

## Selecting Data for Modeling

### The Challenge

Real datasets often have too many columns to work with all at once. You need to:
1. Choose which columns are relevant
2. Focus on features that make sense for prediction
3. Start simple, then iterate with more features

### Viewing Available Columns

```python
import pandas as pd

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 

# View all column names
melbourne_data.columns
```

**Output:**
```
Index(['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',
       'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
       'Longtitude', 'Regionname', 'Propertycount'],
      dtype='object')
```

**What This Shows:**
- 21 total columns in the dataset
- Mix of property characteristics (Rooms, Bathroom, Landsize)
- Location data (Suburb, Lattitude, Longtitude)
- Transaction details (Price, Date, Method)

---

## Handling Missing Values

### Simple Approach: Drop Missing Data

```python
# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)
```

**What This Does:**
- `dropna(axis=0)` - Removes rows with any missing values
- `axis=0` - Operates on rows (axis=1 would be columns)
- Simplest solution for now (better methods covered later)

**Why This Matters:**
- Most ML models can't handle missing values
- Dropping rows is quick but loses data
- Later tutorials cover imputation strategies

---

## Choosing the Prediction Target

### What is a Prediction Target?

The **prediction target** is the variable you want to predict. It's what the model will learn to estimate.

**Convention:** The target is called **y** (lowercase)

### Selecting the Target

```python
y = melbourne_data.Price
```

**Two Ways to Select a Column:**
1. **Dot notation:** `dataframe.ColumnName`
2. **Bracket notation:** `dataframe['ColumnName']`

**Result:**
- `y` is now a Pandas **Series** (one-dimensional data)
- Contains all the house prices from the dataset
- This is what we're trying to predict

### Why "y"?

This comes from mathematical notation:
- In equation form: `y = f(x)`
- `y` = output (what we predict)
- `x` = inputs (what we use to predict)
- `f` = function/model (the pattern we learn)

---

## Selecting Features

### What are Features?

**Features** are the columns used as inputs to the model. They're the characteristics that help predict the target.

**Convention:** Features are called **X** (uppercase)

### Choosing Which Features to Use

For this example, we choose features that intuitively affect house prices:

```python
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
```

**Why These Features?**
- **Rooms** - More rooms typically = higher price
- **Bathroom** - More bathrooms = higher price
- **Landsize** - Larger property = higher price
- **Lattitude/Longtitude** - Location affects price

### Creating the Feature Matrix

```python
X = melbourne_data[melbourne_features]
```

**What This Does:**
- Selects multiple columns at once using a list
- Creates a DataFrame with only the chosen features
- Each row is a house, each column is a feature

**Important Syntax:**
- Double brackets `[[...]]` when selecting multiple columns
- Returns a DataFrame (2D structure)
- Single brackets for one column returns a Series (1D)

### Reviewing Your Features

```python
# View summary statistics
X.describe()
```

```python
# View first few rows
X.head()
```

**Example Output:**
```
   Rooms  Bathroom  Landsize  Lattitude  Longtitude
1      2       1.0     156.0   -37.8079    144.9934
2      3       2.0     134.0   -37.8093    144.9944
4      4       1.0     120.0   -37.8072    144.9941
6      3       2.0     245.0   -37.8024    144.9993
7      2       1.0     256.0   -37.8060    144.9954
```

**Why Review?**
- Verify features loaded correctly
- Check for unexpected values
- Understand the data you're working with
- Catch potential issues early

---

## Building Your Model

### The Modeling Process

Every machine learning workflow follows these steps:

1. **Define** - Choose model type and parameters
2. **Fit** - Train the model on data (learn patterns)
3. **Predict** - Use the model to make predictions
4. **Evaluate** - Check how accurate predictions are

### Step 1: Define the Model

```python
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)
```

**Breaking It Down:**

**Import the Model:**
- `from sklearn.tree` - scikit-learn's decision tree module
- `import DecisionTreeRegressor` - The specific model class

**Create Model Instance:**
- `DecisionTreeRegressor()` - Creates a new decision tree
- `random_state=1` - Ensures reproducible results

**About random_state:**
- Decision trees have randomness in training
- Setting `random_state` gives same results every time
- Any number works (1, 42, 100, etc.)
- Good practice for reproducibility

### Step 2: Fit the Model

```python
# Fit model
melbourne_model.fit(X, y)
```

**What Happens Here:**
- `.fit(X, y)` - Train the model
- `X` - Feature data (inputs)
- `y` - Target data (outputs)
- Model learns patterns: "When Rooms=3 and Landsize=200, Price tends to be around $X"

**This is the "Learning" Step:**
- Algorithm analyzes the data
- Finds patterns between features and target
- Builds internal decision rules
- Creates the trained model

**Return Value:**
```
DecisionTreeRegressor(random_state=1)
```
This confirms the model was trained successfully.

---

## Making Predictions

### Using the Trained Model

```python
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
```

**Output:**
```
Making predictions for the following 5 houses:
   Rooms  Bathroom  Landsize  Lattitude  Longtitude
1      2       1.0     156.0   -37.8079    144.9934
2      3       2.0     134.0   -37.8093    144.9944
4      4       1.0     120.0   -37.8072    144.9941
6      3       2.0     245.0   -37.8024    144.9993
7      2       1.0     256.0   -37.8060    144.9954

The predictions are
[1035000. 1465000. 1600000. 1876000. 1636000.]
```

### Understanding Predictions

**What `.predict()` Does:**
- Takes feature data as input
- Applies learned patterns
- Returns predicted prices

**Example Interpretation:**
- House 1 (2 rooms, 1 bath, 156 sqm) → Predicted price: $1,035,000
- House 4 (4 rooms, 1 bath, 120 sqm) → Predicted price: $1,600,000

**Important Note:**
- Here we're predicting on training data (for demonstration)
- In practice, you predict on NEW houses
- Later lessons cover proper validation

---

## Complete Code Workflow

### Full Working Example

```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Step 1: Load the data
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# Step 2: Handle missing values
melbourne_data = melbourne_data.dropna(axis=0)

# Step 3: Select the prediction target (y)
y = melbourne_data.Price

# Step 4: Choose features (X)
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

# Step 5: Define the model
melbourne_model = DecisionTreeRegressor(random_state=1)

# Step 6: Fit (train) the model
melbourne_model.fit(X, y)

# Step 7: Make predictions
predictions = melbourne_model.predict(X.head())
print("Predictions:", predictions)
```

### The Pipeline Visualized

```
Raw Data (CSV)
      ↓
Load with Pandas
      ↓
Clean Data (drop NaN)
      ↓
Split into X (features) and y (target)
      ↓
Define Model (DecisionTreeRegressor)
      ↓
Fit Model (learn patterns)
      ↓
Trained Model
      ↓
Make Predictions
      ↓
Predicted Prices
```

---

## Key Concepts

### Scikit-Learn (sklearn)

**What It Is:**
- Most popular Python library for machine learning
- Provides tools for modeling, preprocessing, evaluation
- Consistent API across different model types
- Written as `sklearn` in code

**Why It's Popular:**
- Easy to use
- Well-documented
- Efficient implementation
- Wide variety of algorithms

### Model Definition vs. Fitting

**Definition (Creating the Model):**
```python
model = DecisionTreeRegressor(random_state=1)
```
- Creates an "empty" model
- Specifies the type and parameters
- Hasn't learned anything yet

**Fitting (Training the Model):**
```python
model.fit(X, y)
```
- Teaches the model using data
- Model learns patterns
- Now ready to make predictions

**Analogy:**
- **Definition** = Buying a blank notebook
- **Fitting** = Writing notes in the notebook from class

### Features vs. Target

| Aspect | Features (X) | Target (y) |
|--------|-------------|-----------|
| **What** | Input variables | Output variable |
| **Purpose** | Used to make predictions | What we want to predict |
| **Convention** | Uppercase X | Lowercase y |
| **Type** | DataFrame (multiple columns) | Series (single column) |
| **Example** | Rooms, Bathroom, Landsize | Price |

### Data Types in Pandas

**DataFrame:**
- 2-dimensional table
- Multiple rows and columns
- Like an Excel spreadsheet
- Used for features (X)

**Series:**
- 1-dimensional array
- Single column of data
- Like a single column from Excel
- Used for target (y)

---

## Important Terminology

### Decision Tree Regressor

**Regressor:**
- Predicts continuous numerical values
- Examples: prices, temperatures, scores
- Contrast with "Classifier" (predicts categories)

**Decision Tree:**
- Model type that makes decisions with yes/no questions
- Creates a tree structure
- Easy to interpret
- Can capture non-linear patterns

### Random State

**Purpose:**
- Controls randomness in model training
- Ensures reproducible results
- Different numbers give same model quality

**When to Use:**
- Always set it in practice
- Makes debugging easier
- Allows consistent comparisons

### Fitting

**Also Called:**
- Training
- Learning
- Teaching

**What It Means:**
- Process of learning from data
- Adjusting model parameters
- Capturing patterns

---

## Best Practices Learned

### ✓ Do's

1. **Always check your data** - Use `.head()` and `.describe()` before modeling
2. **Handle missing values** - Can't build models with NaN
3. **Set random_state** - Ensures reproducible results
4. **Start simple** - Use few features first, add more later
5. **Verify features** - Make sure they make intuitive sense
6. **Check predictions** - Look at a few examples to sanity-check

### ✗ Don'ts

1. **Don't skip data exploration** - Understand data before modeling
2. **Don't use all features blindly** - More isn't always better
3. **Don't forget to fit** - Can't predict without training first
4. **Don't predict on nothing** - Need feature data to make predictions
5. **Don't assume accuracy** - Will learn to evaluate in next lessons

---

## Common Patterns to Remember

### Loading and Preparing Data

```python
import pandas as pd
data = pd.read_csv('filepath.csv')
data = data.dropna(axis=0)  # Handle missing values
```

### Selecting Target and Features

```python
y = data.TargetColumn
features = ['Feature1', 'Feature2', 'Feature3']
X = data[features]
```

### Building and Training Model

```python
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=1)
model.fit(X, y)
```

### Making Predictions

```python
predictions = model.predict(X)
```

---

## What We Haven't Covered Yet

**Important Topics for Later:**
- How to evaluate model accuracy (next lesson!)
- How to handle missing values properly
- How to choose the best features
- How to prevent overfitting
- How to tune model parameters
- How to predict on truly new data

**Why We Predict on Training Data Here:**
- Just for demonstration purposes
- Shows how `.predict()` works
- Not a proper evaluation (coming next!)

---

## Practical Tips

### Debugging Common Issues

**Error: "X has different number of features"**
- Solution: Make sure X has same columns during fit and predict

**Error: "Input contains NaN"**
- Solution: Use `.dropna()` or handle missing values

**Error: "y should be a 1d array"**
- Solution: Select single column for y, not multiple

**Predictions seem wrong:**
- Check: Did you fit the model first?
- Check: Are you using the right features?
- Check: Is your data clean?

---

## Key Takeaways

✓ **ML workflow is systematic** - Define, Fit, Predict, Evaluate  
✓ **Features are inputs (X)** - What we use to predict  
✓ **Target is output (y)** - What we want to predict  
✓ **Scikit-learn is standard** - Industry-standard ML library  
✓ **Decision trees are interpretable** - Easy to understand how they work  
✓ **Always set random_state** - Ensures reproducible results  
✓ **Data preparation is crucial** - Clean data before modeling  
✓ **Start simple, iterate** - Begin with few features, expand later

---

## Next Steps

After building your first model:

1. **Evaluate accuracy** - How good are the predictions?
2. **Improve the model** - Try different features or parameters
3. **Validate properly** - Test on data the model hasn't seen
4. **Compare models** - Try different algorithms
5. **Deploy** - Use the model for real predictions

**Next Lesson:** Model Validation - Learning how to properly evaluate your model's performance

---

## 🔗 Related Resources

- [Previous Notes: Exercise - Explore Your Data](./exercise-explore-your-data-notes.md)
- [Kaggle Tutorial: Your First ML Model](https://www.kaggle.com/dansbecker/your-first-machine-learning-model)
- [scikit-learn DecisionTreeRegressor Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [Pandas Selecting Data](https://pandas.pydata.org/docs/user_guide/indexing.html)

---

## 📄 License

These are personal learning notes. Original course content by Dan Becker on Kaggle.

---

## 📝 Personal Reflection

**What I Learned:**
- How to structure a complete ML workflow from data to predictions
- The importance of separating features (X) and target (y)
- How to use scikit-learn to build models
- That building a model is surprisingly straightforward with the right tools
- The value of starting simple before adding complexity

**Challenges Faced:**
- Understanding the difference between defining and fitting a model
- Remembering to use double brackets for multiple columns
- Knowing when to use DataFrame vs. Series

**What Made Sense:**
- The systematic workflow (Define → Fit → Predict)
- Why we separate X and y
- The intuition behind choosing features
- How `.predict()` applies learned patterns

**What I'm Excited to Learn Next:**
- How to measure if my predictions are actually good
- Better ways to handle missing values
- How to prevent the model from memorizing training data
- Advanced feature selection techniques

---

**⭐ Successfully built my first machine learning model!**
