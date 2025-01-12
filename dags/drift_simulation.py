from airflow.operators.python_operator import PythonOperator
from airflow import DAG
from datetime import datetime, timedelta
import mlflow
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

spark = SparkSession.builder.getOrCreate()

def train_model(train_data_path='data/WA_Fn-UseC_-Telco-Customer-Churn.csv', new_data_path=''):
    path = train_data_path

    schema = [
        ("customerID", StringType()),
        ("gender", StringType()),
        ("SeniorCitizen", StringType()), #
        ("Partner", StringType()), #
        ("Dependents", StringType()), #
        ("tenure", IntegerType()),
        ("PhoneService", StringType()), #
        ("MultipleLines", StringType()),
        ("InternetService", StringType()),
        ("OnlineSecurity", StringType()),
        ("OnlineBackup", StringType()),
        ("DeviceProtection", StringType()),
        ("TechSupport", StringType()),
        ("StreamingTV", StringType()),
        ("StreamingMovies", StringType()),
        ("Contract", StringType()),
        ("PaperlessBilling", StringType()), #
        ("PaymentMethod", StringType()),
        ("MonthlyCharges", DoubleType()),
        ("TotalCharges", DoubleType()),
        ("Churn", StringType()) #
    ]
    schemaST = StructType([StructField(c, t, True) for c, t in schema])
    df = spark.read.csv(path, header=True, schema=schemaST)

    if new_data_path != '':
        new_df = spark.read.csv(new_data_path, header=True, schema=schemaST)
        df = df.union(new_df)

    df = df.filter(~(df.TotalCharges.isNull() | isnan(df.TotalCharges)))

    numericalCols = []
    categoricalCols = []
    for c, t in schema:
        if t == StringType():
            categoricalCols.append(c)
            continue
        numericalCols.append(c)

    categoricalCols.remove('customerID')
    categoricalCols.remove('Churn')

    label_indexer = StringIndexer(inputCol="Churn", outputCol="label")
    numAssembler = VectorAssembler(inputCols=numericalCols, outputCol="numFeat")
    scaler = StandardScaler(inputCol="numFeat", outputCol="scnumFeat")
    indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categoricalCols]
    encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_vec") for col in categoricalCols]
    final_features = ["scnumFeat"] + [col + "_vec" for col in categoricalCols]
    assembler_final = VectorAssembler(inputCols=final_features, outputCol="final_features")
    lr = LogisticRegression(featuresCol="final_features", labelCol="label")

    pipeline = Pipeline(stages= 
        [label_indexer] + indexers + encoders + [
        numAssembler, scaler,
        assembler_final,
        lr
    ])

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    model = pipeline.fit(train)
    predictions = model.transform(test)
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
    roc_auc = evaluator.evaluate(predictions)
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = accuracy_evaluator.evaluate(predictions)
    log_mlflow(roc_auc, accuracy)

def log_mlflow(roc, accuracy):
    mlflow.start_run()
    mlflow.log_metric("area under roc", roc)
    mlflow.log_metric("accuracy", accuracy)
    # mlflow.log_param("param1", "value")
    mlflow.end_run()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow drift simulation")
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def detect_data_drift(**kwargs):
    # Simulate loading training and new data
    train_data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    new_data = pd.read_csv("new_data.csv")
    
    drift_detected = False
    for col in train_data.columns:
        if abs(train_data[col].mean() - new_data[col].mean()) > 0.1:
            drift_detected = True
            break

    if drift_detected:
        return "retrain_model"
    return "end_pipeline"

def retrain_model(**kwargs):
    mlflow.start_run()
    train_model()
    mlflow.end_run()

# Define DAG
with DAG(
    "data_drift_pipeline",
    default_args=default_args,
    description="A pipeline to detect data drift and retrain model",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    
    # Task 1: Detect data drift
    detect_drift = PythonOperator(
        task_id="detect_data_drift",
        python_callable=detect_data_drift,
    )
    
    # Task 2: Retrain the model
    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
    )
    
    # Task 3: End pipeline (no drift detected)
    end_pipeline = PythonOperator(
        task_id="end_pipeline",
        python_callable=lambda: print("No drift detected"),
    )
    
    # Define task dependencies
    detect_drift >> [retrain, end_pipeline]