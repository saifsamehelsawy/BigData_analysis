# E-Commerce User Behavior Analysis & Prediction

## 📌 Project Overview
This project analyzes e-commerce user behavior data from a multi-category online store (November 2019 dataset). The goal is to clean, explore, visualize, and predict user actions (`view`, `cart`, `purchase`) using a machine learning model (XGBoost).  

It leverages **Apache Spark** for big data processing, **Pandas** & **Seaborn/Matplotlib** for visualization, and **Scikit-learn** + **XGBoost** for predictive modeling.

---

## 📝 Dataset
- File: `2019-Nov.csv`
- Columns include:
  - `event_time`: Timestamp of user action
  - `event_type`: Type of action (`view`, `cart`, `purchase`)
  - `product_id`, `category_id`, `category_code`, `brand`, `price`, `user_id`, `user_session`
- Size: Large dataset (~millions of rows)

---

## 🎯 Project Objectives
1. **Data Cleaning**: Remove missing values, duplicates, and incorrect data.
2. **Exploratory Data Analysis (EDA)**:
   - Missing & unique values
   - Event type counts
   - Top products, brands, users
   - Price statistics
   - Hourly activity patterns
3. **Visualization**:
   - Price distribution
   - Top brands (horizontal bar chart)
   - Top product categories (boxplot & pie chart)
   - Hourly activity
   - Correlation heatmap
   - KDE plot for price
4. **Data Monitoring**:
   - Track Spark job execution and log summaries
5. **Machine Learning**:
   - Predict `eventType` using features like price, brand, category, and hour
   - Use XGBoost Classifier
   - Evaluate model performance

---

## 🛠️ Tech Stack
- **Big Data Processing**: Apache Spark (PySpark)
- **Data Analysis**: Pandas, Numpy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost
- **Python Version**: 3.8+
- **Additional Libraries**: datetime, random

---

## ⚡ Project Structure

### 1️⃣ Import Libraries
All necessary libraries for Spark, ML, and visualization are imported.

### 2️⃣ Initialize Spark Session
Create a Spark session for distributed data processing.

### 3️⃣ Load Raw Data
Load CSV dataset into Spark DataFrame and show sample records.

### 4️⃣ Data Cleaning & Transformation (Initial)
- Drop rows with critical missing values
- Convert columns to correct types (`Timestamp`, `Integer`, `Float`, `Long`)
- Rename columns for consistency
- Write intermediate cleaned CSV

### 5️⃣ Further Cleaning (Remove Duplicates & Fill NA)
- Remove duplicate rows
- Fill missing values in `brand`, `categoryCode`, `price`, `userId`
- Filter rows with valid `price`, `productId`, `userId`, `eventType`
- Save final cleaned dataset

### 6️⃣ Exploratory Data Analysis (EDA)
- Missing & unique values summary
- Event type counts
- Top viewed products
- Price statistics & expensive products
- Top users
- Hourly activity
- Top brands and average prices

### 7️⃣ Visualization
- Price distribution (bar chart)
- Top brands (horizontal bar chart)
- Top 5 product categories (boxplot & pie chart)
- Hourly activity pattern (line chart)
- Correlation heatmap
- Price distribution KDE plot

### 8️⃣ Job Monitoring
- Function `monitored_job()` logs Spark job execution with status, error, start/end times, and duration.
- Tracks:
  - Top brands by count
  - Total record count
  - Writing cleaned data

### 9️⃣ Machine Learning Model (XGBoost)
- Encode categorical variables (`brand`, `categoryCode`, `eventType`)
- Split dataset into train/test (70/30)
- Train XGBoost Classifier to predict `eventType`
- Evaluate model accuracy

---

## 💻 How to Run

1. **Clone repository**:
```bash
git clone <repo_url>
cd <repo_folder>
