# Exercise: Explore Your Data - Notes

[![Course](https://img.shields.io/badge/Course-Intro%20to%20ML-blue)](https://www.kaggle.com/learn/intro-to-machine-learning)
[![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF)](https://www.kaggle.com)
[![Exercise](https://img.shields.io/badge/Type-Hands--On%20Exercise-success)](https://www.kaggle.com)

**📚 Course:** Intro to Machine Learning (Kaggle)  
**👨‍🎓 Completed by:** Dharmi Sapariya  
**📖 Topic:** Hands-On Data Exploration Practice  
**📅 Date:** February 25, 2026  
**🎯 Exercise:** Part 2 - Basic Data Exploration  
**✅ Status:** All Steps Completed Successfully

---

## 📑 Table of Contents

- [Overview](#overview)
- [Dataset Information](#dataset-information)
- [Step 1: Loading Data](#step-1-loading-data)
- [Step 2: Review The Data](#step-2-review-the-data)
- [Statistical Analysis](#statistical-analysis)
- [Key Findings](#key-findings)
- [Complete Code Summary](#complete-code-summary)
- [Insights and Observations](#insights-and-observations)
- [What I Learned](#what-i-learned)

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

### Task
Read the Iowa data file into a Pandas DataFrame called `home_data`.

### My Solution

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

  LandContour Utilities  ... PoolArea PoolQC Fence MiscFeature MiscVal MoSold
0         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2
1         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      5
2         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      9
3         Lvl    AllPub  ...        0    NaN   NaN         NaN       0      2
4         Lvl    AllPub  ...        0    NaN   NaN         NaN       0     12

  YrSold  SaleType  SaleCondition  SalePrice
0   2008        WD         Normal     208500
1   2007        WD         Normal     181500
2   2008        WD         Normal     223500
3   2006        WD        Abnorml     140000
4   2008        WD         Normal     250000

[5 rows x 81 columns]
```

**Key Observations from head():**
- Data loaded successfully
- Multiple feature types visible (numerical and categorical)
- Some NaN (missing) values present
- Each row represents a house with unique ID
- 81 total columns as expected

### Additional Exploration

I went beyond the required task and explored more:

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

---

## Step 2: Review The Data

### Task
Use the command to view summary statistics of the data, then answer questions about average lot size and newest home age.

### My Solution - Part 1: View Statistics

```python
# Print summary statistics
print(home_data.describe())
```

### Understanding .describe()

The `.describe()` method provides statistical summaries for all numerical columns:

**What It Shows:**
- `count` - Number of non-null values
- `mean` - Average value
- `std` - Standard deviation (spread of data)
- `min` - Minimum value
- `25%` - First quartile (25th percentile)
- `50%` - Median (middle value, 50th percentile)
- `75%` - Third quartile (75th percentile)
- `max` - Maximum value

### Statistical Summary Output

```
                Id   MSSubClass  LotFrontage        LotArea  OverallQual
count  1460.000000  1460.000000  1201.000000    1460.000000  1460.000000
mean    730.500000    56.897260    70.049958   10516.828082     6.099315
std     421.610009    42.300571    24.284752    9981.264932     1.382997
min       1.000000    20.000000    21.000000    1300.000000     1.000000
25%     365.750000    20.000000    59.000000    7553.500000     5.000000
50%     730.500000    50.000000    69.000000    9478.500000     6.000000
75%    1095.250000    70.000000    80.000000   11601.500000     7.000000
max    1460.000000   190.000000   313.000000  215245.000000    10.000000

       OverallCond    YearBuilt  YearRemodAdd   MasVnrArea   BsmtFinSF1  ...
count  1460.000000  1460.000000   1460.000000  1452.000000  1460.000000  ...
mean      5.575342  1971.267808   1984.865753   103.685262   443.639726  ...
std       1.112799    30.202904     20.645407   181.066207   456.098091  ...
min       1.000000  1872.000000   1950.000000     0.000000     0.000000  ...
25%       5.000000  1954.000000   1967.000000     0.000000     0.000000  ...
50%       5.000000  1973.000000   1994.000000     0.000000   383.500000  ...
75%       6.000000  2000.000000   2004.000000   166.000000   712.250000  ...
max       9.000000  2010.000000   2010.000000  1600.000000  5644.000000  ...

        WoodDeckSF  OpenPorchSF  EnclosedPorch    3SsnPorch  ScreenPorch
count  1460.000000  1460.000000    1460.000000  1460.000000  1460.000000
mean     94.244521    46.660274      21.954110     3.409589    15.060959
std     125.338794    66.256028      61.119149    29.317331    55.757415
min       0.000000     0.000000       0.000000     0.000000     0.000000
25%       0.000000     0.000000       0.000000     0.000000     0.000000
50%       0.000000    25.000000       0.000000     0.000000     0.000000
75%     168.000000    68.000000       0.000000     0.000000     0.000000
max     857.000000   547.000000     552.000000   508.000000   480.000000

          PoolArea       MiscVal       MoSold       YrSold      SalePrice
count  1460.000000   1460.000000  1460.000000  1460.000000    1460.000000
mean      2.758904     43.489041     6.321918  2007.815753  180921.195890
std      40.177307    496.123024     2.703626     1.328095   79442.502883
min       0.000000      0.000000     1.000000  2006.000000   34900.000000
25%       0.000000      0.000000     5.000000  2007.000000  129975.000000
50%       0.000000      0.000000     6.000000  2008.000000  163000.000000
75%       0.000000      0.000000     8.000000  2009.000000  214000.000000
max     738.000000  15500.000000    12.000000  2010.000000  755000.000000

[8 rows x 38 columns]
```

---

## Statistical Analysis

### Understanding the Data Through Statistics

Let me break down key statistics from the output:

### LotArea (Property Size)

```
count:    1460.000000
mean:  10516.828082 sq ft
std:    9981.264932 sq ft
min:    1300.000000 sq ft
25%:    7553.500000 sq ft
50%:    9478.500000 sq ft
75%:   11601.500000 sq ft
max:  215245.000000 sq ft
```

**Key Insights:**
- Average lot size is about 10,517 sq ft
- High standard deviation (9,981) indicates large variation
- Smallest lot: 1,300 sq ft
- Largest lot: 215,245 sq ft (HUGE outlier!)
- Median (9,478) is less than mean - suggests right-skewed distribution
- 75% of properties are under 11,601 sq ft

### YearBuilt

```
count:    1460.000000
mean:   1971.267808
std:      30.202904
min:    1872.000000
25%:    1954.000000
50%:    1973.000000
75%:    2000.000000
max:    2010.000000
```

**Key Insights:**
- Oldest house built in 1872 (154 years old!)
- Newest house built in 2010
- Average construction year: 1971
- Median year: 1973
- 25% of houses built before 1954
- 75% of houses built before 2000

### SalePrice (Target Variable)

```
count:    1460.000000
mean:  180921.195890
std:    79442.502883
min:    34900.000000
25%:   129975.000000
50%:   163000.000000
75%:   214000.000000
max:   755000.000000
```

**Key Insights:**
- Average price: $180,921
- High variation: $79,442 standard deviation
- Cheapest house: $34,900
- Most expensive: $755,000
- Median price: $163,000 (below average - right-skewed)
- 50% of houses between $129,975 and $214,000

### OverallQual (Quality Rating 1-10)

```
mean:       6.099315
std:        1.382997
min:        1.000000
50%:        6.000000
max:       10.000000
```

**Key Insights:**
- Average quality rating: 6.1 out of 10
- Most houses rated around 5-7
- Full range from 1 (poor) to 10 (excellent)

---

## Key Findings

### Questions Answered

The exercise required answering two specific questions:

#### Question 1: What is the average lot size (rounded to nearest integer)?

```python
avg_lot_size = round(home_data["LotArea"].mean())
print(avg_lot_size)
```

**Output:** `10517`

**Answer:** 10,517 square feet

**How I Got This:**
1. Selected the "LotArea" column using bracket notation
2. Used `.mean()` method to calculate the average
3. Used `round()` function to round to nearest integer
4. Stored result in `avg_lot_size` variable

**What This Tells Us:**
- The typical Iowa house in this dataset sits on about 10,517 sq ft
- That's roughly a quarter acre (43,560 sq ft per acre)
- This is a suburban/small town lot size

#### Question 2: As of today, how old is the newest home?

```python
newest_home_age = 2026 - home_data["YearBuilt"].max()
print(newest_home_age)
```

**Output:** `16`

**Answer:** 16 years old

**How I Got This:**
1. Found the maximum value in "YearBuilt" column using `.max()`
2. This gave us 2010 (the newest house's construction year)
3. Subtracted from current year (2026)
4. Result: 2026 - 2010 = 16 years

**What This Tells Us:**
- The newest house in the dataset was built in 2010
- That house is now 16 years old
- No houses built after 2010 in this dataset
- This raises important questions about data freshness

---

## Complete Code Summary

### Full Working Code

```python
import pandas as pd

# Step 1: Load the data
iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)

# View first few rows
print(home_data.head())

# View detailed column information
print(home_data.info())

# View statistical summary
print(home_data.describe())

# Step 2: Answer specific questions

# Calculate average lot size (rounded)
avg_lot_size = round(home_data["LotArea"].mean())
print(f"Average Lot Size: {avg_lot_size} sq ft")

# Calculate age of newest home
newest_home_age = 2026 - home_data["YearBuilt"].max()
print(f"Age of Newest Home: {newest_home_age} years")
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

### Data Quality Discoveries

**1. Missing Values Present**

From the `.info()` output, several columns have missing data:
- LotFrontage: 1,201 non-null (259 missing)
- Alley: 91 non-null (1,369 missing - 94%!)
- MasVnrType: 588 non-null (872 missing)
- MasVnrArea: 1,452 non-null (8 missing)

**Why This Matters:**
- Models can't handle missing values
- Will need imputation strategy later
- Alley column might not be useful (too many missing)

**2. Outliers Detected**

Several features show extreme maximum values:
- LotArea max: 215,245 sq ft (20x the average!)
- SalePrice max: $755,000 (4x the average)
- These could skew model predictions

**3. Skewed Distributions**

Many features show right-skewed distributions (mean > median):
- LotArea: mean 10,517 > median 9,478
- SalePrice: mean $180,921 > median $163,000

**Why This Matters:**
- A few very large/expensive properties pull the average up
- Median is often more representative
- May need data transformations for modeling

### Data Age Considerations

**The Newest House Problem:**

The newest house being 16 years old (built 2010) raises questions:

**Two Possible Explanations:**

1. **No New Construction**
   - Area stopped building houses
   - Economic factors (2008 recession aftermath?)
   - Could still be valid for that area

2. **Old Dataset**
   - Data collected around 2010
   - Houses built after 2010 not included
   - Model predictions might be outdated

**Impact on Model Trust:**

If Explanation #1:
- Model still valid for that specific area
- But limited to areas with similar patterns
- Might not generalize to growing markets

If Explanation #2:
- Model uses outdated prices
- Housing market changed since 2010
- Predictions may not reflect current market

**How to Investigate:**

Could look at the `YrSold` column to see when sales occurred:
```python
print(home_data["YrSold"].describe())
```

If all sales are 2006-2010, likely Explanation #2.

---

## What I Learned

### Technical Skills Acquired

✅ **Loading Data**
- Used `pd.read_csv()` to load external CSV file
- Understood file paths in Kaggle notebooks
- Verified successful loading with `.head()`

✅ **Data Exploration Methods**
- `.head()` - Quick visual inspection
- `.info()` - Column data types and missing values
- `.describe()` - Statistical summaries

✅ **Statistical Calculations**
- Calculated mean with `.mean()`
- Found maximum with `.max()`
- Rounded numbers with `round()`

✅ **Column Selection**
- Used bracket notation: `data["ColumnName"]`
- Understood when to use brackets vs dot notation

✅ **Data Quality Assessment**
- Identified missing values
- Spotted outliers
- Recognized skewed distributions

### Conceptual Understanding

**1. Why Explore Before Modeling**

Data exploration revealed:
- Missing values need handling
- Outliers might affect model
- Data age might impact predictions
- Distribution shapes inform feature engineering

**2. Statistics Tell Stories**

Numbers aren't just numbers:
- Mean vs median reveals skewness
- Standard deviation shows variability
- Min/max identify outliers
- Quartiles show distribution shape

**3. Domain Knowledge Matters**

Understanding what features mean:
- 215,245 sq ft lot = unusual (investigate why)
- Houses from 1872 = historical homes (different pricing)
- 94% missing "Alley" data = most houses don't have alleys (normal)

**4. Questions Lead to Insights**

Simple questions (average lot size, newest home age) led to:
- Understanding typical property size
- Discovering potential data age issues
- Thinking about model applicability

### Best Practices Learned

**✓ Always Explore First**
- Never build models blindly
- Statistics reveal data characteristics
- Visual inspection catches obvious issues

**✓ Check Multiple Aspects**
- Not just `.head()` - also `.info()` and `.describe()`
- Each method reveals different insights
- Comprehensive view prevents surprises

**✓ Question the Data**
- Why is newest house 16 years old?
- Why so many missing "Alley" values?
- Is 215,245 sq ft lot a data error?

**✓ Document Findings**
- Note unusual patterns
- Record missing value counts
- Save insights for feature engineering

---

## Comparing to Tutorial

### Tutorial (Melbourne) vs Exercise (Iowa)

| Aspect | Melbourne (Tutorial) | Iowa (Exercise) |
|--------|---------------------|-----------------|
| **Country** | Australia | USA |
| **Rows** | ~13,000 | 1,460 |
| **Columns** | 21 | 81 |
| **Target** | Price | SalePrice |
| **Years** | Various | 2006-2010 |
| **My Role** | Follow along | Independent work |

**What This Taught Me:**
- Same techniques work on different datasets
- Must adapt to different column names
- Each dataset has unique challenges
- Real understanding = can apply independently

---

## Critical Thinking Questions

### Questions Raised During Exercise

**Q1: Why does Alley have 94% missing values?**
- **A:** Most houses don't have alley access (this is normal, not a data problem)

**Q2: Is the 215,245 sq ft lot real or an error?**
- **A:** Could be a large rural property, or data entry error - would investigate further

**Q3: Should I remove outliers?**
- **A:** Not yet - they might be legitimate. Decision trees can handle outliers well.

**Q4: How will missing values affect my model?**
- **A:** Most algorithms can't handle NaN - will learn handling strategies in later lessons

**Q5: Is 16-year-old newest house a problem?**
- **A:** Depends on use case. For historical analysis, fine. For current predictions, concerning.

---

## Common Mistakes Avoided

### ❌ Mistakes I Didn't Make

**1. Not Exploring Data**
- Many beginners jump straight to modeling
- I took time to understand the data first
- This prevents surprises later

**2. Ignoring Missing Values**
- Saw that several columns have NaN
- Noted this for future handling
- Won't get caught by error messages later

**3. Accepting Statistics Blindly**
- Questioned why mean ≠ median
- Investigated extreme values
- Understood what numbers actually mean

**4. Using Wrong Column**
- Iowa uses "SalePrice" not "Price"
- Checked actual column names
- Avoided naming errors

### ✅ Good Practices I Followed

**1. Comprehensive Exploration**
- Used multiple methods (.head, .info, .describe)
- Got complete picture of data
- Ready for next steps

**2. Clear Code Comments**
- Documented what each section does
- Future me will understand the code
- Good professional habit

**3. Verification Steps**
- Printed results to verify correctness
- Checked that answers make sense
- Caught errors early

**4. Asking "Why?"**
- Didn't just calculate, but understood
- Questioned unusual patterns
- Developed critical thinking

---

## Next Steps After This Exercise

### What Comes Next

Having explored the data, I'm now ready to:

1. **Select Features** - Choose which columns to use for modeling
2. **Handle Missing Values** - Decide strategy for NaN values
3. **Build First Model** - Create a simple baseline
4. **Validate Model** - Check prediction accuracy
5. **Iterate and Improve** - Refine based on results

### Preparation for Model Building

This exploration prepared me by:
- Understanding data structure
- Identifying data quality issues
- Knowing variable distributions
- Having baseline statistics for comparison

---

## Personal Reflection

### What Went Well

✅ Successfully loaded and explored Iowa housing dataset  
✅ Correctly calculated required statistics  
✅ Went beyond requirements (used .info() too)  
✅ Identified important data quality issues  
✅ Asked critical questions about data age  
✅ Understood why statistics matter  

### Challenges Overcome

**Challenge 1:** Remembering syntax
- **Solution:** Referred to tutorial notes
- **Lesson:** Documentation is valuable

**Challenge 2:** Understanding standard deviation
- **Solution:** Looked at context (9,981 std dev vs 10,517 mean = high variation)
- **Lesson:** Statistics need interpretation

**Challenge 3:** Deciding what's "normal"
- **Solution:** Used domain knowledge (most houses don't have alleys)
- **Lesson:** Context matters in data analysis

### Key Takeaway

**Main Insight:**
Data exploration isn't just running commands - it's detective work. Every statistic tells a story. Every missing value has a reason. Every outlier deserves investigation. Taking time to understand data before modeling is not optional - it's essential.

**Quote to Remember:**
> "Garbage in, garbage out. Understand your data before you model it."

---

## 🔗 Related Resources

- [Previous: Basic Data Exploration Tutorial](./02-basic-data-exploration-notes.md)
- [Next: Your First Machine Learning Model](./04-your-first-machine-learning-model-notes.md)
- [Kaggle Exercise](https://www.kaggle.com/code/dharmisapariya/exercise-explore-your-data)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

## 📄 License

These are personal learning notes. Original exercise by Kaggle.

---

**✅ Data Exploration Completed - Ready to Build First Model!**

**💡 Core Learning:** Understanding your data through exploration is the foundation of all successful machine learning projects. Statistics reveal patterns, missing values indicate real-world complexity, and questioning unusual findings leads to deeper insights.
