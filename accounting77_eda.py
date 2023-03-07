# Databricks notebook source
#### test the push
tutorial_path = "/FileStore/tables/BERT" 

# COMMAND ----------

from collections import namedtuple
import numpy as np
import pandas as pd
from datasets import load_dataset
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType, LongType
import pyspark.sql.functions as func
from pyspark.sql.functions import col
from pyspark.sql import SparkSession

# COMMAND ----------

### function from dataset to dataframes
def dataset_to_dataframe(data_name: str):
    """
    this code is to transform the data to dataframe
    """
    spark_data = namedtuple("spark_data","train test labels")
    
    # define spark schemas
    single_label_schema = StructType([StructField("text",StringType(), False),
                              StructField("label",LongType(),False)
                              ])
    labels_schema = StructType([StructField("idx",IntegerType(), False),
                              StructField("label",StringType(),False)
                              ])
    
    # start the data
    dataset = load_dataset(data_name)
    train_pd = dataset['train'].to_pandas()
    test_pd = dataset['test'].to_pandas()
    train_pd = train_pd.sample(frac=1).reset_index(drop=True)
    test_pd = test_pd.sample(frac=1).reset_index(drop =True)
    
    train = spark.createDataFrame(train_pd, schema = single_label_schema)
    test = spark.createDataFrame(test_pd, schema = single_label_schema)
    
    idx_and_labels = dataset['train'].features['label'].names
    
    # the below is a must for transformer model
    id2label = [(idx,label) for idx, label in enumerate(idx_and_labels)]
    labels = spark.createDataFrame(id2label, schema = labels_schema)
    
    return spark_data(train, test, labels)

# COMMAND ----------

banking77_df = dataset_to_dataframe("banking77")


# COMMAND ----------

# write them to delta format
banking77_df.train.write.format("parquet").mode("overwrite").save("dbfs:/FileStore/tables/BERT/banking77_train")
banking77_df.test.write.format("parquet").mode("overwrite").save("dbfs:/FileStore/tables/BERT/banking77_test")
banking77_df.labels.write.format("parquet").mode("overwrite").save("dbfs:/FileStore/tables/BERT/banking77_labels")

# COMMAND ----------

## read from parquet
banking77_train_df = spark.read.parquet("dbfs:/FileStore/tables/BERT/banking77_train")
banking77_test_df = spark.read.parquet("dbfs:/FileStore/tables/BERT/banking77_test")
banking77_labels_df = spark.read.parquet("dbfs:/FileStore/tables/BERT/banking77_labels")


# COMMAND ----------

# Write the tables to disk. 
# train_dbfs_path = f"{tutorial_path}/banking77_train"
# test_dbfs_path = f"{tutorial_path}/banking77_test"

# banking77_train = banking77_df.train.write.parquet(train_dbfs_path, mode="overwrite")
# banking77_test = banking77_df.test.write.parquet(test_dbfs_path, mode="overwrite")# Write out processed data to parquet to the driver.


# COMMAND ----------


