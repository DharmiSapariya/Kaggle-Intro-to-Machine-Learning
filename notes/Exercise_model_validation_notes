# Exercise: Model Validation - Notes

[![Course](https://img.shields.io/badge/Course-Intro%20to%20ML-blue)](https://www.kaggle.com/learn/intro-to-machine-learning)
[![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF)](https://www.kaggle.com)
[![Exercise](https://img.shields.io/badge/Type-Hands--On%20Exercise-success)](https://www.kaggle.com)

**📚 Course:** Intro to Machine Learning (Kaggle)  
**👨‍🎓 Completed by:** Dharmi Sapariya  
**📖 Topic:** Model Validation - Testing Model Performance  
**📅 Date:** February 25, 2026  
**🎯 Exercise:** Part 4 - Model Validation Exercise  
**✅ Status:** All Steps Completed Successfully

---

## 📑 Table of Contents

- [Overview](#overview)
- [The Problem We're Solving](#the-problem-were-solving)
- [Step 1: Split Your Data](#step-1-split-your-data)
- [Step 2: Specify and Fit Model](#step-2-specify-and-fit-model)
- [Step 3: Make Predictions](#step-3-make-predictions)
- [Step 4: Calculate Mean Absolute Error](#step-4-calculate-mean-absolute-error)
- [Key Insights](#key-insights)
- [Complete Code Summary](#complete-code-summary)
- [What I Learned](#what-i-learned)

---

## Overview

This exercise teaches proper model validation - how to test if your model will work well on new, unseen data. Previously, I predicted on training data and got perfect results. Now I'll learn why that's misleading and how to properly evaluate models.

**Key Concepts Practiced:**
- Train-test splitting
- Validation data
- Mean Absolute Error (MAE)
- Proper model evaluation

---

## The Problem We're Solving

### Recap: Previous Exercise Results

In the last exercise, I got these results:

```python
print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())
```

**Output:**
```
First in-sample predictions: [208500. 181500. 223500. 140000. 250000.]
Actual target values for those homes: [208500, 181500, 223500, 140000, 250000]
```

**Perfect match!** But this is misleading because:
- Model was trained on this data
- Already "seen" these houses
- Memorized the answers
- Won't work well on new houses

### The Real Question

**Will the model work on houses it hasn't seen during training?**

This is what we'll test in this exercise.

---

## Step 1: Split Your Data

### Task
Split the data into training and validation sets using `train_test_split`.

### My Solution

```python
# Step 1: Split your data
from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(
    X, y, random_state=1
)

# Check your answer
step_1.check()
```

**Result:** ✅ Correct

### Understanding the Split

**What This Function Does:**

```python
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
```

**Returns 4 Datasets:**

1. **train_X** - Features for training (75% of data by default)
2. **val_X** - Features for validation (25% of data)
3. **train_y** - Target values for training (75%)
4. **val_y** - Target values for validation (25%)

**Parameters:**
- `X` - All feature data
- `y` - All target data
- `random_state=1` - Ensures reproducible split

**Default Split:**
- Training: 75% of the data (1,095 houses)
- Validation: 25% of the data (365 houses)

### Why Split the Data?

**Training Set:**
- Used to teach the model
- Model learns patterns from this data
- Larger portion (75%)

**Validation Set:**
- Used to test the model
- Model has NEVER seen this during training
- Smaller portion (25%)
- Simulates "new" houses

**The Key Principle:**
> "Never test on data you trained on"

### Visualizing the Split

```
Original Data (1,460 houses)
         |
         ↓
   train_test_split
         |
    _____|_____
   |           |
Training     Validation
(75%)        (25%)
1,095        365 houses
houses
   |           |
   ↓           ↓
Model         Test
Learns        Performance
Here          Here
```

---

## Step 2: Specify and Fit Model

### Task
Create a DecisionTreeRegressor and train it on the TRAINING data only.

### My Solution

```python
# Step 2: Specify and fit model

# Specify the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit iowa_model with the training data
iowa_model.fit(train_X, train_y)

# Check your answer
step_2.check()
```

**Result:** ✅ Correct

**Output During Check:**
```
[186500. 184000. 130000.  92000. 164500. 220000. 335000. 144152. 215000. 262000.]
[186500. 184000. 130000.  92000. 164500. 220000. 335000. 144152. 215000. 262000.]
```

### Critical Difference

**Previous Exercise:**
```python
iowa_model.fit(X, y)  # Trained on ALL data (1,460 houses)
```

**This Exercise:**
```python
iowa_model.fit(train_X, train_y)  # Trained on TRAINING data only (1,095 houses)
```

**Why This Matters:**
- Model only learns from training set
- Validation set remains "unseen"
- Can now properly test performance
- Simulates real-world scenario

### What Happens During Training

```
Training Data (1,095 houses)
         ↓
   Model Analysis
         ↓
   Learns Patterns:
   - "More rooms → higher price"
   - "Newer houses → higher price"
   - "Larger lot → higher price"
   - etc.
         ↓
   Trained Model
   (Ready to predict)
```

**Important Note:**
The model has NO INFORMATION about the 365 validation houses. This is intentional!

---

## Step 3: Make Predictions

### Task
Make predictions on the validation data (data the model hasn't seen).

### My Solution

```python
# Step 3: Make predictions on validation data
val_predictions = iowa_model.predict(val_X)

# Check your answer
step_3.check()
```

**Result:** ✅ Correct

### Understanding Validation Predictions

**What We're Doing:**
```python
val_predictions = iowa_model.predict(val_X)
```

**Breaking It Down:**
- `iowa_model.predict()` - Use trained model to predict
- `val_X` - Validation features (365 houses the model NEVER saw)
- `val_predictions` - Predicted prices for these unseen houses

**This is the Real Test:**
- Model must predict on completely new data
- Can't memorize (hasn't seen these houses)
- Shows true predictive ability
- Mimics real-world usage

### Comparing Predictions

**Training Predictions (from earlier):**
```python
# Model predicts on data it was trained on
train_predictions = iowa_model.predict(train_X)
# Result: Perfect or near-perfect accuracy
```

**Validation Predictions (now):**
```python
# Model predicts on data it has NEVER seen
val_predictions = iowa_model.predict(val_X)
# Result: More realistic accuracy (to be measured)
```

---

## Step 4: Calculate Mean Absolute Error

### Task
Calculate the Mean Absolute Error (MAE) on validation predictions.

### My Solution

```python
from sklearn.metrics import mean_absolute_error

# Calculate MAE on validation data
val_mae = mean_absolute_error(val_y, val_predictions)

# Uncomment to see the value
#print(val_mae)

# Check your answer
step_4.check()
```

**Result:** ✅ Correct

### Understanding Mean Absolute Error (MAE)

**What is MAE?**

Mean Absolute Error measures the average difference between predicted and actual values.

**Formula:**
```
MAE = Average of |Predicted - Actual|
```

**Example Calculation:**

If we have 3 houses:

| House | Predicted | Actual | Error | Absolute Error |
|-------|-----------|--------|-------|----------------|
| 1 | $200,000 | $210,000 | -$10,000 | $10,000 |
| 2 | $150,000 | $145,000 | +$5,000 | $5,000 |
| 3 | $180,000 | $190,000 | -$10,000 | $10,000 |

```
MAE = ($10,000 + $5,000 + $10,000) / 3 = $8,333
```

**Interpretation:**
"On average, our predictions are off by $8,333"

### Why Use Absolute Values?

Without absolute values:
- Errors can cancel out
- -$10,000 and +$10,000 = $0 average (misleading!)
- Absolute values prevent cancellation
- Shows true magnitude of errors

### The Code Explained

```python
val_mae = mean_absolute_error(val_y, val_predictions)
```

**Parameters:**
- `val_y` - Actual prices from validation set (365 values)
- `val_predictions` - Predicted prices from model (365 values)

**Returns:**
- Single number representing average error
- Lower is better
- Same units as target (dollars in this case)

### What Makes a "Good" MAE?

**Context Matters:**

For house prices around $150,000-$200,000:
- MAE of $10,000 = 5-7% error (pretty good!)
- MAE of $50,000 = 25-33% error (not great)
- MAE of $100,000 = 50-67% error (poor)

**No Universal "Good" Value:**
- Depends on the problem domain
- Depends on price ranges
- Depends on business requirements
- Compare to baseline or other models

---

## Key Insights

### Training vs Validation Performance

**Important Discovery:**

When I checked the predictions during Step 2, I saw:
```
Predictions: [186500. 184000. 130000. ...]
Actuals:     [186500. 184000. 130000. ...]
```

These matched perfectly because they were from the TRAINING set (the model had seen this data).

**The Real Test:**
Validation predictions on UNSEEN data will have some error - and that's normal and expected!

### Why Validation Matters

**Without Validation:**
- Only test on training data
- Get misleadingly good results
- Model might have memorized data
- Fail on real-world predictions

**With Validation:**
- Test on unseen data
- Get realistic performance estimates
- Detect overfitting
- Confident about real-world performance

### The Validation Workflow

```
1. Split Data
   ↓
2. Train on Training Set
   ↓
3. Predict on Validation Set
   ↓
4. Calculate Error (MAE)
   ↓
5. Improve Model
   ↓
6. Repeat 2-5
   ↓
7. Final Model for Production
```

---

## Complete Code Summary

### Full Working Code

```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load data
iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)

# Select target and features
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 
                   'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# STEP 1: Split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# STEP 2: Specify and train model (on training data only!)
iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)

# STEP 3: Make predictions on validation data
val_predictions = iowa_model.predict(val_X)

# STEP 4: Calculate validation MAE
val_mae = mean_absolute_error(val_y, val_predictions)

print(f"Validation MAE: ${val_mae:,.0f}")
```

### The Complete Pipeline

```
Raw Data (1,460 houses)
         ↓
Select Features & Target
         ↓
Split into Train/Val (75%/25%)
         ↓
         ├─── Training (1,095) ──→ Train Model
         |
         └─── Validation (365) ──→ Test Model
                                    ↓
                              Calculate MAE
                                    ↓
                              Evaluate Performance
```

---

## What I Learned

### Technical Skills

✅ **Train-Test Splitting**
- How to use `train_test_split()`
- Understanding the 75/25 default split
- Importance of `random_state` for reproducibility

✅ **Proper Model Training**
- Train only on training data
- Keep validation data separate
- Never let model see validation during training

✅ **Model Evaluation**
- How to calculate Mean Absolute Error
- Using `mean_absolute_error()` function
- Interpreting MAE in context

✅ **Validation Workflow**
- The complete process from split to evaluation
- Why each step matters
- Order of operations

### Conceptual Understanding

**1. Why Perfect Training Accuracy is Suspicious**
- In last exercise: Perfect predictions on training data
- This exercise: More realistic errors on validation data
- Lesson: Training performance ≠ real performance

**2. The Purpose of Validation Data**
- Simulates "new" houses model hasn't seen
- Provides realistic performance estimate
- Catches overfitting early
- Builds confidence in model

**3. Train-Test Contamination**
- Never use validation data for training
- Keep the sets completely separate
- Otherwise results are meaningless
- This is a cardinal rule in ML

**4. MAE as a Metric**
- Simple, interpretable measure of error
- Same units as target variable
- Easy to explain to non-technical people
- Good starting metric

### Critical Insights

**💡 Key Realization:**
"The model I built in the previous exercise looked perfect, but I was testing it wrong. By predicting on training data, I was essentially asking the model to repeat what it had memorized. Validation data shows the true story."

**⚠️ Common Mistake Avoided:**
Many beginners test on training data and think their model is amazing. This exercise taught me to always split data and test properly.

**🎯 Best Practice Learned:**
Always split data BEFORE training. Never touch validation data until you're ready to test.

---

## Comparing Exercise 3 vs Exercise 4

| Aspect | Exercise 3 (Previous) | Exercise 4 (This) |
|--------|----------------------|-------------------|
| **Training Data** | All 1,460 houses | 1,095 houses (75%) |
| **Testing Data** | Same 1,460 houses | 365 new houses (25%) |
| **Predictions** | On training data | On validation data |
| **Results** | Perfect accuracy | Realistic accuracy |
| **Problem** | Misleading | Honest evaluation |
| **Lesson** | How to build model | How to validate model |

---

## Understanding the Difference

### Previous Exercise (Wrong Way)

```python
# Train on ALL data
iowa_model.fit(X, y)

# Test on SAME data (wrong!)
predictions = iowa_model.predict(X)

# Result: Unrealistically perfect
```

### This Exercise (Right Way)

```python
# Split data first
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Train on TRAINING data only
iowa_model.fit(train_X, train_y)

# Test on VALIDATION data (unseen!)
val_predictions = iowa_model.predict(val_X)

# Result: Realistic performance
val_mae = mean_absolute_error(val_y, val_predictions)
```

---

## Real-World Analogy

**Bad Approach (Exercise 3):**
- Teacher gives students practice problems
- Students memorize the answers
- Test uses THE SAME problems
- Everyone gets 100% (but learned nothing)

**Good Approach (Exercise 4):**
- Teacher gives students practice problems
- Students learn the concepts
- Test uses DIFFERENT problems
- Scores reflect true understanding

**Validation data = Different test problems**

---

## Questions Answered

### Q: Why not just use all data for training?

**A:** Because then you have no way to test if the model will work on new data. You need held-out data to validate performance.

### Q: Why 75/25 split?

**A:** It's a common default that balances:
- Enough training data to learn well (75%)
- Enough validation data to test reliably (25%)

Can be adjusted based on data size and needs.

### Q: What if I accidentally train on validation data?

**A:** Your validation results become meaningless. The whole point is that validation data is unseen during training.

### Q: Is validation data "wasted"?

**A:** No! It serves a crucial purpose: honest evaluation. Without it, you have no idea if your model works.

---

## Common Mistakes to Avoid

### ❌ Don't Do This

```python
# WRONG: Training on all data
model.fit(X, y)
predictions = model.predict(X)  # Predicting on training data
```

### ❌ Don't Do This

```python
# WRONG: Using validation data for training
model.fit(val_X, val_y)
```

### ❌ Don't Do This

```python
# WRONG: Splitting after training
model.fit(X, y)  # Already trained on everything!
train_X, val_X = train_test_split(X, y)  # Too late
```

### ✅ Do This

```python
# RIGHT: Split first, then train on training only
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)
mae = mean_absolute_error(val_y, val_predictions)
```

---

## What's Next

### Building on This Foundation

Now that I understand proper validation, I can:

1. **Compare Different Models**
   - Try various algorithms
   - Compare validation MAE
   - Choose the best performer

2. **Tune Parameters**
   - Adjust model settings
   - See impact on validation error
   - Find optimal configuration

3. **Add Features**
   - Try different feature combinations
   - Measure impact on validation
   - Keep what improves performance

4. **Avoid Overfitting**
   - Next lesson: Underfitting vs Overfitting
   - Learn to balance model complexity
   - Use validation to detect overfitting

---

## Personal Reflection

### What Went Well

✅ All 4 steps completed on first try  
✅ Understood the importance of data splitting  
✅ Grasped why previous perfect results were misleading  
✅ Learned proper model validation workflow  
✅ Can now evaluate models honestly  

### Challenges Overcome

**Challenge 1:** Understanding why we split data
- **Solved:** Realized it simulates real-world scenario

**Challenge 2:** Remembering to use train_X not X
- **Solved:** Understood the point is to keep validation unseen

**Challenge 3:** Interpreting MAE value
- **Solved:** Learned it's context-dependent

### Key Takeaway

**Previous Exercise:** "Look, my model is perfect!"  
**This Exercise:** "Actually, let me test that properly..."

The biggest lesson: **Honest evaluation is more valuable than impressive-looking results.** A model that performs well on validation data is trustworthy. A model that only performs well on training data is suspicious.

---

## 🔗 Related Resources

- [Previous: Exercise - Your First ML Model](./exercise-your-first-machine-learning-model-notes.md)
- [Kaggle Exercise: Model Validation](https://www.kaggle.com/code/dharmisapariya/exercise-model-validation)
- [Next: Underfitting and Overfitting](https://www.kaggle.com/dansbecker/underfitting-and-overfitting)
- [scikit-learn train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

---

## 📄 License

These are personal learning notes. Original exercise by Kaggle.

---

## 🎓 Key Concepts Mastered

✅ **Train-Test Split** - Dividing data into training and validation sets  
✅ **Validation Data** - Unseen data for honest evaluation  
✅ **Mean Absolute Error** - Measuring prediction accuracy  
✅ **Proper Evaluation** - Never test on training data  
✅ **Realistic Performance** - Getting honest metrics  

---

**✅ Validation Workflow Mastered - Ready to Learn About Overfitting!**

**💡 Core Insight:** The difference between looking good on training data and actually working on new data is the foundation of all machine learning evaluation. This exercise taught me to always be skeptical of perfect results and to validate properly.
