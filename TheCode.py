# ================================================================
# 1️⃣ IMPORT LIBRARIES
# ================================================================
from pyspark.sql import SparkSession
from pyspark.sql import *
from pyspark.sql.functions import col, when, isnull, count, countDistinct, avg, desc, udf, isnan, hour, split
from pyspark.sql.types import *
from pyspark.sql import functions as f
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import random
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ================================================================
# 2️⃣ INITIALIZE SPARK SESSION
# ================================================================
spark = SparkSession.builder \
    .appName("eCommerce behavior data from multi category store") \
    .getOrCreate()

# ================================================================
# 3️⃣ LOAD RAW DATA
# ================================================================
df = spark.read.csv("2019-Nov.csv", header=True, inferSchema=True)
df.show(20)

# ================================================================
# 4️⃣ DATA CLEANING & TRANSFORMATION (INITIAL)
# ================================================================
df_cleaned = df.dropna(subset=["event_time", "event_type", "product_id", "user_id"])
df_cleaned = df_cleaned.withColumn("event_time", col("event_time").cast(TimestampType())) \
                       .withColumn("product_id", col("product_id").cast(IntegerType())) \
                       .withColumn("category_id", col("category_id").cast(LongType())) \
                       .withColumn("price", col("price").cast(FloatType())) \
                       .withColumn("user_id", col("user_id").cast(IntegerType()))

# Rename columns
df_cleaned = df_cleaned.withColumnRenamed("event_time", "eventTime") \
                       .withColumnRenamed("event_type", "eventType") \
                       .withColumnRenamed("product_id", "productId") \
                       .withColumnRenamed("category_id", "categoryId") \
                       .withColumnRenamed("category_code", "categoryCode") \
                       .withColumnRenamed("user_id", "userId") \
                       .withColumnRenamed("user_session", "userSession")

df_cleaned.printSchema()
df_cleaned.show(20, truncate=False)

# Write cleaned data (initial)
(df_cleaned.coalesce(1)
    .write.mode("overwrite")
    .option("header", "true")
    .csv("cleaned_ecommerce_behavior_single"))
print("Cleaned e-commerce behavior data written as a single CSV in 'cleaned_ecommerce_behavior_single' directory")

# ================================================================
# 5️⃣ FURTHER CLEANING (REMOVE DUPLICATES & FILL NA)
# ================================================================
file_name = "part-0000.csv"
df = spark.read.option("header", True).csv(file_name)
df_cleaned = df.dropDuplicates()
df_cleaned = df_cleaned.fillna({
    "brand": "Realme",
    "categoryCode": "electronics.smartphone",
    "price": 0.0,
    "userId": 0
})

df_cleaned = df_cleaned.withColumn("eventTime", col("eventTime").cast(TimestampType())) \
                       .withColumn("productId", col("productId").cast(IntegerType())) \
                       .withColumn("categoryId", col("categoryId").cast(LongType())) \
                       .withColumn("price", col("price").cast(FloatType())) \
                       .withColumn("userId", col("userId").cast(IntegerType()))

# Filter relevant rows
df_cleaned = df_cleaned.filter(
    (col("price") > 0) &
    (col("productId").isNotNull()) &
    (col("userId") != 0) &
    (col("eventType").isin(["view", "cart", "purchase"])) &
    (col("brand") != "")
)

df_cleaned.printSchema()
df_cleaned.show(20)

# Write final cleaned data
df_cleaned.repartition(1) \
          .write.mode("overwrite") \
          .option("header", "true") \
          .csv("cleaned_ecommerce_behavior_single_file")
print("Cleaned e-commerce behavior data saved successfully.")

# ================================================================
# 6️⃣ EXPLORATORY DATA ANALYSIS (EDA)
# ================================================================

# 6.1 Missing & unique values
df_cleaned.select([count(when(col(c).isNull() | (col(c) == ""), c)).alias(c) for c in df.columns]).show()
df_cleaned.select([countDistinct(col(c)).alias(c) for c in df.columns]).show()

# 6.2 Event type counts
df_cleaned.groupBy("eventType").count().orderBy("count", ascending=False).show()

# 6.3 Top viewed products
df_cleaned.filter(col("eventType") == "view") \
          .groupBy("productId").count().orderBy(col("count").desc()).show(10)

# 6.4 Price stats & expensive products
df_cleaned.select("price").describe().show()
df_cleaned.orderBy(col("price").desc()).select("productId", "brand", "price").show(10)

# 6.5 Top users
df_cleaned.groupBy("userId").count().orderBy(col("count").desc()).show(10)

# 6.6 Hourly activity
df_cleaned.withColumn("hour", hour(col("eventTime"))) \
          .groupBy("hour").count().orderBy("hour").show()

# 6.7 Top brands & avg price
df_cleaned.filter(col("eventType") == "view") \
          .groupBy("brand").count().orderBy(col("count").desc()).show(10)
df_cleaned.groupBy("brand").agg(avg("price").alias("avg_price")) \
          .orderBy(col("avg_price").desc()).show(10)
print("EDA Done")

# ================================================================
# 7️⃣ VISUALIZATION
# ================================================================

# 7.1 Price distribution
price_bins = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000)]
price_labels = [f"${start}-{end}" for start, end in price_bins]
price_distribution = df_cleaned.select(
    when(col("price") < 50, price_labels[0])
     .when((col("price") >= 50) & (col("price") < 100), price_labels[1])
     .when((col("price") >= 100) & (col("price") < 200), price_labels[2])
     .when((col("price") >= 200) & (col("price") < 500), price_labels[3])
     .when((col("price") >= 500) & (col("price") < 1000), price_labels[4])
     .otherwise(price_labels[5]).alias("price_range")
).groupBy("price_range").count().orderBy("price_range").collect()

ranges = [row["price_range"] for row in price_distribution]
counts = [row["count"] for row in price_distribution]

plt.figure(figsize=(12, 6))
bars = plt.bar(ranges, counts, color='#1f77b4', edgecolor='black')
plt.title("Distribution of Product Prices", fontsize=14, pad=20)
plt.xlabel("Price Range ($)", fontsize=12)
plt.ylabel("Number of Products", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# 7.2 Top brands (horizontal bar chart)
top_brands_df = df_cleaned.filter(col("eventType") == "view") \
                          .groupBy("brand") \
                          .agg(count("*").alias("count")) \
                          .orderBy(col("count").desc()) \
                          .limit(20) \
                          .toPandas()

plt.figure(figsize=(12, 7))
ax = sns.barplot(x=top_brands_df['count'], y=top_brands_df['brand'], palette='viridis', orient='h')
plt.title('Top 20 Most Viewed Brands', fontsize=16, pad=20)
plt.xlabel('Number of Views', fontsize=12)
plt.ylabel('Brand', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=8)
for i, count_val in enumerate(top_brands_df['count']):
    ax.text(count_val + 0.1, i, f"{count_val:,}", va='center', fontsize=10)
plt.tight_layout()
plt.show()

# 7.3 Top 5 product categories (boxplot & pie)
top_categories = df_cleaned.withColumn("main_category", split(col("categoryCode"), "\.")[0]) \
                           .groupBy("main_category").agg(count("*").alias("count")) \
                           .orderBy(col("count").desc()).limit(5).collect()

top_5_categories = [row["main_category"] for row in top_categories]

# Boxplot
category_price_data = df_cleaned.withColumn("main_category", split(col("categoryCode"), "\.")[0]) \
                                .filter(col("main_category").isin(top_5_categories)) \
                                .select("main_category", "price").toPandas()
plt.figure(figsize=(10, 6))
sns.boxplot(y="price", x="main_category", data=category_price_data, palette="pastel", width=0.6, showfliers=False)
plt.title('Price Distribution by Top 5 Product Categories', fontsize=14, pad=20)
plt.ylabel('Price ($)', fontsize=12)
plt.xlabel('Main Category', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)
medians = category_price_data.groupby('main_category')['price'].median()
for i, category in enumerate(top_5_categories):
    median_val = medians[category]
    plt.text(i, median_val + 20, f'${median_val:.0f}', ha='center', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
plt.tight_layout()
plt.show()

# Pie chart
category_dist = df_cleaned.withColumn("main_category", split(col("categoryCode"), "\\.")[0]) \
                          .groupBy("main_category").count() \
                          .orderBy(col("count").desc()).limit(5).toPandas()
plt.figure(figsize=(8, 8))
plt.pie(category_dist['count'], labels=category_dist['main_category'], autopct='%1.1f%%', startangle=90,
        colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
plt.title('Top 5 Product Categories Distribution', pad=20)
plt.tight_layout()
plt.show()

# 7.4 Hourly activity
hourly_counts = df_cleaned.withColumn("hour", hour("eventTime")) \
                          .groupBy("hour").count().orderBy("hour").toPandas()
plt.figure(figsize=(12, 6))
plt.plot(hourly_counts['hour'], hourly_counts['count'], marker='s', linestyle='-', color='#ff7f0e', linewidth=2)
plt.title('Hourly Activity Pattern', fontsize=14)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Number of Events', fontsize=12)
plt.xticks(range(24))
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()

# 7.5 Correlation heatmap
df_num = df_cleaned.select(["price", "userId", "productId", "categoryId"])
corr_matrix = df_num.toPandas().corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("E-commerce Metrics Correlation", pad=20, fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.show()

# 7.6 KDE plot for price
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df_cleaned.toPandas(), x="price", fill=True, color="#af1fb4", alpha=0.5, linewidth=2, bw_method=0.2)
plt.title("Product Price Distribution (Smooth Curve)", fontsize=16)
plt.xlabel("Price ($)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.3)
sns.despine()
plt.show()

# ================================================================
# 8️⃣ JOB MONITORING FUNCTION
# ================================================================
job_logs = []

def monitored_job(job_name, action):
    start_time = datetime.now()
    status = "Success"
    error_message = ""
    try:
        print(f"\n▶ Running job: {job_name}...")
        action()
    except Exception as e:
        status = "Failed"
        error_message = str(e)
        print(f"Job '{job_name}' failed.")
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    job_logs.append({
        "Job Name": job_name,
        "Status": status,
        "Error": error_message if error_message else "None",
        "Start Time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
        "End Time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
        "Duration (s)": round(duration, 2)
    })

# Execute Spark Jobs
monitored_job("Top Brands by Count", lambda: df_cleaned.groupBy("brand").count().orderBy(col("count").desc()).show(20))
monitored_job("Record Count", lambda: print(f"Total Records: {df_cleaned.count()}"))
monitored_job("Write Cleaned Data", lambda: df_cleaned.repartition(1).write.mode("overwrite").option("header", "true").csv("output_cleaned_data"))

# Display the job logs
print("\n📄 Job Execution Summary:")
for log in job_logs:
    print(f"- Job: {log['Job Name']}")
    print(f"  Status: {log['Status']}")
    print(f"  Error: {log['Error']}")
    print(f"  Start Time: {log['Start Time']}")
    print(f"  End Time: {log['End Time']}")
    print(f"  Duration: {log['Duration (s)']} seconds\n")

# ================================================================
# 9️⃣ MACHINE LEARNING MODEL (XGBOOST)
# ================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

df_with_hour = df.withColumn("hour", hour(col("eventTime")))
df_pd = df_with_hour.select("price", "brand", "categoryCode", "hour", "eventType").toPandas()

# Encode categorical variables
le_event = LabelEncoder()
le_brand = LabelEncoder()
le_cat = LabelEncoder()
df_pd['eventType'] = le_event.fit_transform(df_pd['eventType'])
df_pd['brand'] = le_brand.fit_transform(df_pd['brand'])
df_pd['categoryCode'] = le_cat.fit_transform(df_pd['categoryCode'])
df_pd['price'] = df_pd['price'].astype(float)

# Features & target
features = ['price', 'brand', 'categoryCode', 'hour']
target = 'eventType'
X = df_pd[features]
y = df_pd[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Train XGBoost Classifier
xgb_model = XGBClassifier(random_state=SEED)
xgb_model.fit(X_train, y_train)

# Predictions & accuracy
y_pred = xgb_model.predict(X_test)
y_test_int = y_test.astype(int)
acc = accuracy_score(y_test_int, y_pred)
print("Approximate Accuracy:", round(acc * 100, 2), "%")
