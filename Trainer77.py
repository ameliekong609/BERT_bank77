# Databricks notebook source
import json
from time import perf_counter
from argparse import Namespace
from typing import List, Tuple, Dict
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
from pynvml import *
import mlflow
from mlflow.tracking import MlflowClient
import pickle
from pathlib import Path
from sys import version_info

from datasets import load_dataset, DatasetDict
import numpy as np
import mlflow

from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import logistic
import torch
from transformers import (AutoConfig,
                          AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          EarlyStoppingCallback, 
                          EvalPrediction, 
                          DataCollatorWithPadding,
                          pipeline,
                          TrainingArguments, 
                          Trainer)    

# COMMAND ----------

tutorial_path = "/FileStore/tables/BERT" 

train_dbfs_path = f"{tutorial_path}/banking77_train"
test_dbfs_path = f"{tutorial_path}/banking77_test"


# COMMAND ----------

# Load the files into transformers compatible Datasets.

## read from parquet
# banking77_train_df = spark.read.parquet("dbfs:/FileStore/tables/BERT/banking77_train")
# banking77_test_df = spark.read.parquet("dbfs:/FileStore/tables/BERT/banking77_test")


train_test = load_dataset("parquet", data_files={"train":f"/dbfs{train_dbfs_path}/*.parquet", "test":f"/dbfs{test_dbfs_path}/*.parquet"})

 

# print(f"train_cnt: {train_test.train.count()}, test_cnt: {train_test.test.count()}, labels_cnt: {train_test.label.count()}")

# COMMAND ----------

### setting up parameters
banking77_labels_df = spark.read.parquet("dbfs:/FileStore/tables/BERT/banking77_labels")
labels = banking77_labels_df.collect()
id2label = {index: row.label for (index, row) in enumerate(labels)} 
label2id = {row.label: index for (index, row) in enumerate(labels)}

# COMMAND ----------

 ### tokenize
model_type ="bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=True)
def tokenize_function(batch):
  """Tokenize input text in batches"""

  return tokenizer(batch["text"], 
                   truncation=True,
                   padding=False,
                   max_length=512)
  

# COMMAND ----------

# The DataCollator will handle dynamic padding of batches during training. See the documentation, 
# https://www.youtube.com/watch?v=-RPeakdlHYo. If not leveraging dynamic padding, this can be removed
data_collator = DataCollatorWithPadding(tokenizer, padding=True)

# The default batch size is 1,000; this can be changed by setting the "batch_size=" parameter
# https://huggingface.co/docs/datasets/process#batch-processing

train_test_tokenized = train_test.map(tokenize_function, batched=True).remove_columns(["text"])

# train_dataset = train_test_tokenized["train"].shuffle(seed=42)
# test_dataset = train_test_tokenized["test"].shuffle(seed=42)

# COMMAND ----------

import numpy as np
import evaluate
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# COMMAND ----------

from transformers import TrainingArguments, Trainer
training_output_dir = "sms_trainer"
training_args = TrainingArguments(output_dir=training_output_dir, evaluation_strategy="epoch") 

# COMMAND ----------

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=77, label2id=label2id, id2label=id2label)

# COMMAND ----------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test_tokenized["train"],
    eval_dataset=train_test_tokenized["test"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)


# COMMAND ----------

import mlflow
from tqdm.auto import tqdm
import torch

pipeline_artifact_name = "BERTpipeline"
class TextClassificationPipelineModel(mlflow.pyfunc.PythonModel):
  
  def load_context(self, context):
    device = 0 if torch.cuda.is_available() else -1
    self.pipeline = pipeline("text-classification", context.artifacts[pipeline_artifact_name], device=device)
    
  def predict(self, context, model_input): 
    texts = model_input[model_input.columns[0]].to_list()
    pipe = tqdm(self.pipeline(texts, truncation=True, batch_size=8), total=len(texts), miniters=10)
    labels = [prediction['label'] for prediction in pipe]
    return pd.Series(labels)

# COMMAND ----------


model_output_dir = "./sms_model"
pipeline_output_dir = "./sms_pipeline"
model_artifact_path = "sms_spam_model"

with mlflow.start_run() as run:
  trainer.train()
  trainer.save_model(model_output_dir)
  pipe = pipeline("text-classification", model=AutoModelForSequenceClassification.from_pretrained(model_output_dir), batch_size=8, tokenizer=tokenizer)
  pipe.save_pretrained(pipeline_output_dir)
  mlflow.pyfunc.log_model(artifacts={pipeline_artifact_name: pipeline_output_dir}, artifact_path=model_artifact_path, python_model=TextClassificationPipelineModel())

# COMMAND ----------

logged_model = "runs:/{run_id}/{model_artifact_path}".format(run_id=run.info.run_id, model_artifact_path=model_artifact_path)

# Load model as a Spark UDF. Override result_type if the model does not return double values.
sms_spam_model_udf = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='string')

test = test_df.select(test_df.text, test_df.label, sms_spam_model_udf(test_df.text).alias("prediction"))
display(test)
