# Databricks notebook source
data_path = "/Volumes/workspace/default/upi_fraud/PS_20174392719_1491204439457_log.csv"

# COMMAND ----------

# MAGIC %pip install graphframes

# COMMAND ----------

from graphframes import GraphFrame

# COMMAND ----------

from pyspark.sql.functions import col, to_timestamp, from_unixtime

# Path to your dataset
data_path = "/Volumes/workspace/default/upi_fraud/PS_20174392719_1491204439457_log.csv"

# 1. Load CSV
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(data_path)

# 2. Rename columns to standard format
df = df.select(
    col("nameOrig").alias("sender"),
    col("nameDest").alias("receiver"),
    col("amount"),
    col("step")
)

# 3. Convert "step" to timestamp
# (step = seconds → convert to timestamp)
df = df.withColumn("timestamp", from_unixtime(col("step")))

# 4. Save as Delta table
df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("transactions_bronze")

# 5. Show data
df.show(5)

# COMMAND ----------

from pyspark.sql.functions import col, from_unixtime

# Load CSV
data_path = "/Volumes/workspace/default/upi_fraud/PS_20174392719_1491204439457_log.csv"

df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(data_path)

# Select + convert step → timestamp (DO NOT KEEP step)
df = df.select(
    col("nameOrig").alias("sender"),
    col("nameDest").alias("receiver"),
    col("amount"),
    from_unixtime(col("step")).alias("timestamp")   # convert here
)

# Save Bronze table
df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("transactions_bronze")

# Check
df.show(5)

# COMMAND ----------

from pyspark.sql.functions import col, to_date

# 1. Read Bronze table
df = spark.read.table("transactions_bronze")

# 2. Remove null values
df = df.dropna(subset=["sender", "receiver", "amount", "timestamp"])

# 3. Remove duplicate rows
df = df.dropDuplicates()

# 4. Create transaction_date column from timestamp
df = df.withColumn("transaction_date", to_date(col("timestamp")))

# 5. Save as Delta table (Silver layer)
df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("transactions_silver")

# 6. Show row count
print("Row count:", df.count())

# COMMAND ----------

from pyspark.sql.functions import col, expr, rand

# 1. Read Silver table
df = spark.read.table("transactions_silver")

# Create list to store scaled copies
dfs = []

# 2 & 3. Duplicate 10 times with transformations
for i in range(10):
    temp_df = df \
        .withColumn(
            "timestamp",
            expr(f"timestamp + interval {i} days")  # shift time
        ) \
        .withColumn(
            "amount",
            col("amount") * (1 + (rand() - 0.5) * 0.1)  # ±5% noise
        )
    
    dfs.append(temp_df)

# 4. Combine all datasets
from functools import reduce
df_scaled = reduce(lambda a, b: a.unionByName(b), dfs)

# 5. Save as Delta table
df_scaled.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("transactions_scaled")

# 6. Print total row count
print("Total rows:", df_scaled.count())

# COMMAND ----------

# Imports
import random
from pyspark.sql.functions import col, lit, unix_timestamp, from_unixtime, rand

# 1. Load scaled dataset
df = spark.read.table("transactions_scaled")

# --------------------------------------------------
# 2. Handle real fraud if present
# --------------------------------------------------
if "isFraud" in df.columns:
    df = df.withColumn("is_fraud", col("isFraud"))
else:
    df = df.withColumn("is_fraud", lit(0))

# --------------------------------------------------
# 3. Get unique accounts safely
# --------------------------------------------------
accounts_df = (
    df.select(col("sender").alias("account"))
      .union(df.select(col("receiver").alias("account")))
      .distinct()
      .limit(5000)
)

accounts = [row["account"] for row in accounts_df.collect()]

# --------------------------------------------------
# 4. Create NOISY fraud patterns
# --------------------------------------------------
fraud_data = []

for _ in range(30):
    ring_size = random.randint(5, 10)
    ring_accounts = random.sample(accounts, ring_size)

    base_time = "2024-01-01 00:00:00"

    for i in range(ring_size):
        sender = ring_accounts[i]
        receiver = ring_accounts[(i + 1) % ring_size]

        # 🔴 Break perfect cycles randomly
        if random.random() < 0.6:
            receiver = random.choice(accounts)

        # 🔴 Vary amounts
        amount = float(random.uniform(500, 10000))

        fraud_data.append({
            "sender": sender,
            "receiver": receiver,
            "amount": amount,
            "timestamp": base_time,
            "is_fraud": 1
        })

    # 🔵 Add noisy normal-like transactions
    for _ in range(random.randint(5, 8)):
        fraud_data.append({
            "sender": random.choice(ring_accounts),
            "receiver": random.choice(accounts),
            "amount": float(random.uniform(100, 5000)),
            "timestamp": base_time,
            "is_fraud": 0   # important → noise
        })

# --------------------------------------------------
# 5. Convert fraud data to DataFrame
# --------------------------------------------------
fraud_df = spark.createDataFrame(fraud_data)

# --------------------------------------------------
# 6. Add realistic timestamps
# --------------------------------------------------
fraud_df = fraud_df.withColumn(
    "timestamp",
    from_unixtime(
        unix_timestamp(col("timestamp")) + (rand() * 3600).cast("int")  # up to 1 hour variation
    )
)

# --------------------------------------------------
# 7. Match schema
# --------------------------------------------------
fraud_df = fraud_df.select("sender", "receiver", "amount", "timestamp", "is_fraud")
df = df.select("sender", "receiver", "amount", "timestamp", "is_fraud")

# --------------------------------------------------
# 8. Append fraud + real data
# --------------------------------------------------
df_final = df.unionByName(fraud_df)

# --------------------------------------------------
# 9. Save final dataset
# --------------------------------------------------
df_final.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("transactions_final")

# --------------------------------------------------
# 10. Verify
# --------------------------------------------------
print("Total rows:", df_final.count())
print("Fraud rows:", df_final.filter(col("is_fraud") == 1).count())

df_final.show(5)

# COMMAND ----------

spark.sql("SHOW TABLES").show()

# COMMAND ----------

from pyspark.sql.functions import col, count, sum

# 1. Read final dataset
df = spark.read.table("transactions_final")

# 2. Create NODES (unique accounts)
nodes = (
    df.select(col("sender").alias("id"))
      .union(df.select(col("receiver").alias("id")))
      .distinct()
)

# 3. Create EDGES (transactions between accounts)
edges = (
    df.groupBy("sender", "receiver")
      .agg(
          count("*").alias("transaction_count"),
          sum("amount").alias("total_amount")
      )
      .withColumnRenamed("sender", "src")
      .withColumnRenamed("receiver", "dst")
)

# 4. Save tables
nodes.write.format("delta").mode("overwrite").saveAsTable("nodes_table")
edges.write.format("delta").mode("overwrite").saveAsTable("edges_table")

# 5. Preview
print("Nodes:")
nodes.show(5)

print("Edges:")
edges.show(5)

# COMMAND ----------

nodes_pd = spark.read.table("nodes_table").toPandas()
edges_pd = spark.read.table("edges_table").toPandas()

# COMMAND ----------

# MAGIC %pip install networkx pyvis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trying smaller graphs

# COMMAND ----------

# Load edges
edges_pd = spark.read.table("edges_table").toPandas()

# Take top edges by transaction count (more meaningful)
edges_sample = edges_pd.sort_values(
    "transaction_count", ascending=False
).head(2000)

# COMMAND ----------

nodes_sample = list(
    set(edges_sample["src"]).union(set(edges_sample["dst"]))
)

# COMMAND ----------

import networkx as nx

G = nx.from_pandas_edgelist(
    edges_sample,
    source="src",
    target="dst",
    edge_attr=["transaction_count", "total_amount"],
    create_using=nx.DiGraph()
)

# COMMAND ----------

degree_dict = dict(G.degree())

# Threshold (tune this)
suspicious_nodes = [node for node, deg in degree_dict.items() if deg > 10]

# COMMAND ----------

from pyvis.network import Network

net = Network(height="600px", width="100%", directed=True)

# Add nodes
for node in G.nodes():
    if node in suspicious_nodes:
        net.add_node(node, label=str(node), color="red")
    else:
        net.add_node(node, label=str(node))

# Add edges
for u, v, data in G.edges(data=True):
    net.add_edge(
        u, v,
        value=data["transaction_count"],
        title=f"Txns: {data['transaction_count']}, Amt: {data['total_amount']}"
    )

# Save graph
net.write_html("fraud_network.html")

# COMMAND ----------

# MAGIC %md
# MAGIC TRying further smaller graphs, the above html did not make much sense

# COMMAND ----------

id="fix1"
# Get top nodes by degree
degree_dict = dict(G.degree())

top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:40]

id="fix2"
subG = G.subgraph(top_nodes)



# COMMAND ----------

id="fix3"
from pyvis.network import Network

net = Network(height="600px", width="100%", directed=True)

for node in subG.nodes():
    net.add_node(node, label=str(node), color="red")

for u, v, data in subG.edges(data=True):
    net.add_edge(u, v)

net.write_html("fraud_subgraph2.html")

# COMMAND ----------

spark.sql("SHOW TABLES").show()

# COMMAND ----------

from pyspark.sql.functions import col, sum as _sum, count as _count

# Load tables
nodes = spark.read.table("nodes_table")
edges = spark.read.table("edges_table")

# ---------------------------------------
# 1. Compute in-degree
in_deg = (
    edges.groupBy("dst")
    .agg(_count("*").alias("in_degree"))
    .withColumnRenamed("dst", "id")
)

# ---------------------------------------
# 2. Compute out-degree
out_deg = (
    edges.groupBy("src")
    .agg(_count("*").alias("out_degree"))
    .withColumnRenamed("src", "id")
)

# ---------------------------------------
# 3. Total transaction count (already aggregated edges)
tx_count = (
    edges.select("src", "transaction_count")
    .groupBy("src")
    .agg(_sum("transaction_count").alias("tx_count_out"))
    .withColumnRenamed("src", "id")
)

tx_count_in = (
    edges.select("dst", "transaction_count")
    .groupBy("dst")
    .agg(_sum("transaction_count").alias("tx_count_in"))
    .withColumnRenamed("dst", "id")
)

# ---------------------------------------
# 4. Total amount
amount_out = (
    edges.groupBy("src")
    .agg(_sum("total_amount").alias("amount_out"))
    .withColumnRenamed("src", "id")
)

amount_in = (
    edges.groupBy("dst")
    .agg(_sum("total_amount").alias("amount_in"))
    .withColumnRenamed("dst", "id")
)

# ---------------------------------------
# 5. Degree
graph_features = nodes \
    .join(in_deg, "id", "left") \
    .join(out_deg, "id", "left") \
    .fillna(0)

graph_features = graph_features.withColumn(
    "degree",
    col("in_degree") + col("out_degree")
)

# ---------------------------------------
# 6. Approx cluster size (based on shared connections)
# (simple proxy: count neighbors)
neighbors_src = edges.groupBy("src").agg(_count("dst").alias("neighbor_count"))
neighbors_dst = edges.groupBy("dst").agg(_count("src").alias("neighbor_count_in"))

neighbors = neighbors_src.withColumnRenamed("src", "id") \
    .join(neighbors_dst.withColumnRenamed("dst", "id"), "id", "outer") \
    .fillna(0)

neighbors = neighbors.withColumn(
    "cluster_size",
    col("neighbor_count") + col("neighbor_count_in")
)

# ---------------------------------------
# 7. Triangle proxy (approx)
# nodes with high mutual connections
triangle_proxy = graph_features.select("id", "degree") \
    .withColumnRenamed("degree", "triangle_count")

# ---------------------------------------
# 8. Combine everything
ml_df = graph_features \
    .join(tx_count, "id", "left") \
    .join(tx_count_in, "id", "left") \
    .join(amount_out, "id", "left") \
    .join(amount_in, "id", "left") \
    .join(neighbors.select("id", "cluster_size"), "id", "left") \
    .join(triangle_proxy, "id", "left") \
    .fillna(0)

# ---------------------------------------
# 9. Final selection
ml_df = ml_df.select(
    "id",
    "degree",
    "tx_count_in",
    "tx_count_out",
    "amount_in",
    "amount_out",
    "cluster_size",
    "triangle_count"
)

# ---------------------------------------
# 10. Save
ml_df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("ml_features")

# Preview
ml_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adding label to features

# COMMAND ----------

from pyspark.sql.functions import col, max as _max

# Load features
ml_df = spark.read.table("ml_features")

# Load original data
tx_df = spark.read.table("transactions_final")

# Create label per node (if any transaction is fraud → node is fraud)
labels = (
    tx_df.groupBy("sender")
    .agg(_max("is_fraud").alias("is_fraud"))
    .withColumnRenamed("sender", "id")
)

labels_in = (
    tx_df.groupBy("receiver")
    .agg(_max("is_fraud").alias("is_fraud_in"))
    .withColumnRenamed("receiver", "id")
)

labels = labels.join(labels_in, "id", "outer").fillna(0)

labels = labels.withColumn(
    "label",
    (col("is_fraud") + col("is_fraud_in") > 0).cast("int")
)

# Join with features
data = ml_df.join(labels.select("id", "label"), "id", "left").fillna(0)
data.show(5)

# COMMAND ----------

data.printSchema()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql.functions import col, when

# Compute class imbalance
fraud_count = data.filter("label = 1").count()
nonfraud_count = data.filter("label = 0").count()

ratio = nonfraud_count / fraud_count

# Add weight
data = data.withColumn(
    "weight",
    when(col("label") == 1, ratio).otherwise(1.0)
)

feature_cols = [
    "degree",
    "tx_count_in",
    "tx_count_out"
    # optionally add back if needed:
    # "amount_in",
    # "amount_out"
    # "cluster_size",
    # "triangle_count"
]
# Assemble features
data_with_features = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
).transform(data)

# Keep weight also
data_with_features = data_with_features.select("id", "features", "label", "weight")

# Split by id
ids = data_with_features.select("id").distinct()
train_ids, test_ids = ids.randomSplit([0.8, 0.2], seed=42)

train_df = data_with_features.join(train_ids, "id")
test_df = data_with_features.join(test_ids, "id")

# Select columns
train_df = train_df.select("features", "label", "weight")
test_df = test_df.select("features", "label", "weight")
# Model
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=30,
    maxDepth=3,
    weightCol="weight"
)

model = rf.fit(train_df)

# Predictions
predictions = model.transform(test_df)
from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array

from pyspark.ml.functions import vector_to_array

from pyspark.sql.functions import col

predictions = predictions.withColumn(

    "prob_array",

    vector_to_array(col("probability"))

)

predictions = predictions.withColumn(

    "prediction_custom",

    (col("prob_array")[1] > 0.1).cast("int")

)


# COMMAND ----------

predictions_clean = predictions.select(
    "features",
    "label",
    "prediction",          # original (double)
    "prediction_custom"    # your threshold version
)

# COMMAND ----------

predictions_clean.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("fraud_predictions")

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"   # ✅ VERY IMPORTANT
)

pr_auc = evaluator.evaluate(predictions)
print("PR-AUC:", pr_auc)

# COMMAND ----------

predictions.groupBy("label", "prediction").count().show()

# COMMAND ----------

data.groupBy("label").count().show()

# COMMAND ----------

dbutils.fs.mkdirs("/Volumes/workspace/default/upi_fraud/mlflow_tmp")

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.ml.functions import vector_to_array
from mlflow.models.signature import infer_signature
import os

mlflow.set_experiment("/Shared/fraud_detection")

os.environ["MLFLOW_DFS_TMP"] = "/Volumes/workspace/default/upi_fraud/mlflow_tmp"

with mlflow.start_run():

    # Params
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("num_trees", 40)
    mlflow.log_param("max_depth", 4)

    # Metrics
    mlflow.log_metric("pr_auc", pr_auc)

    # FIX: convert vector → array
    input_example = train_df.select(
        vector_to_array("features").alias("features")
    ).limit(5)

    pred_sample = model.transform(train_df.limit(5))

    input_pd = input_example.toPandas()
    output_pd = pred_sample.select("prediction").toPandas()

    signature = infer_signature(input_pd, output_pd)

    # Log model ONCE (remove duplicate call)
    mlflow.spark.log_model(
        model,
        "rf_model",
        signature=signature,
        input_example=input_pd
    )

# COMMAND ----------

