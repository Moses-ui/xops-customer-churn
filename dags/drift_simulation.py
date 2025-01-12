from airflow.operators.python_operator import PythonOperator
from airflow import DAG
from datetime import datetime, timedelta
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, NumericType

def get_spark_session():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    # .config("spark.driver.extraJavaOptions", 
    #         "-Dhttp.proxyHost=mlflow -Dhttp.proxyPort=5000") \
    # .config("spark.executor.extraJavaOptions", 
    #         "-Dhttp.proxyHost=mlflow -Dhttp.proxyPort=5000") \
    return spark

DATA_PATH = '/opt/airflow/data/'
TRAIN_DATA_PATH=DATA_PATH + 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
NEW_DATA_PATH=DATA_PATH + 'new_data.csv'
SCHEMA=[
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

EXPERIMENT_NAME = "MLflow drift simulation"
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def read_and_combine_data(schema, train_data_path, new_data_path):
    spark = get_spark_session()
    path = train_data_path
    schemaST = StructType([StructField(c, t, True) for c, t in schema])
    df = spark.read.csv(path, header=True, schema=schemaST)

    if new_data_path != '':
        new_df = spark.read.csv(new_data_path, header=True, schema=schemaST)
        df = df.union(new_df)
    
    return df

def train_model(new_data_path, schema=SCHEMA, train_data_path=TRAIN_DATA_PATH):
    from pyspark.sql.functions import isnan
    from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
    df = read_and_combine_data(schema, train_data_path, new_data_path)
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
    log_mlflow(roc_auc, accuracy, model)

def log_mlflow(roc, accuracy, model):
    import mlflow
    import mlflow.spark
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run()
    mlflow.log_metric("area under roc", roc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.spark.log_model(spark_model=model, artifact_path="artifacts", registered_model_name="Churn_Pipeline_Model")
    # mlflow.log_param("param1", "value")
    mlflow.end_run()

def calculate_distribution(df, bin_col):
    from pyspark.sql.functions import col, count, lit
    total_count = df.count()
    return df.groupBy(bin_col).agg(count(lit(1)).alias("count")).withColumn(
        "percentage", col("count") / total_count
    )

def detect_data_drift(**kwargs):
    from pyspark.sql.functions import col, lit, when
    from pyspark.ml.feature import StringIndexer
    spark = get_spark_session()
    schemaST = StructType([StructField(c, t, True) for c, t in SCHEMA])
    expected_df = spark.read.csv(TRAIN_DATA_PATH, header=True, schema=schemaST)
    actual_df = spark.read.csv(NEW_DATA_PATH, header=True, schema=schemaST)

    num_bins = 10
    drift_detected = False
    for c in expected_df.columns:
        if isinstance(expected_df.schema[c].dataType, NumericType):
            processed_column = c
        else:
            indexer = StringIndexer(inputCol=c, outputCol="indexed_" + c)
            expected_df = indexer.fit(expected_df).transform(expected_df)
            actual_df = indexer.fit(actual_df).transform(actual_df)
            processed_column = "indexed_" + c

        expected_df = expected_df.withColumn(
            c,
            when(col(processed_column).isNull(), lit("null"))
            .otherwise((col(processed_column) / lit(num_bins)).cast("int")),
        )
        actual_df = actual_df.withColumn(
            c,
            when(col(processed_column).isNull(), lit("null"))
            .otherwise((col(processed_column) / lit(num_bins)).cast("int")),
        )

        expected_dist = calculate_distribution(expected_df, c)
        actual_dist = calculate_distribution(actual_df, c)

        psi_df = expected_dist.join(
            actual_dist, c, how="outer"
        ).select(
            col(c),
            col("percentage").alias("expected_percentage"),
            col("actual_percentage"),
        )

        psi_df = psi_df.fillna({"expected_percentage": 1e-6, "actual_percentage": 1e-6})
        psi_df = psi_df.withColumn(
            "psi",
            (col("expected_percentage") - col("actual_percentage"))
            * (col("expected_percentage") / col("actual_percentage")).log(),
        )
        total_psi = psi_df.selectExpr("sum(psi) as total_psi").collect()[0]["total_psi"]
        if total_psi > 0.1:
            drift_detected = True
            break

    if drift_detected:
        return "retrain_model"
    return "end_pipeline"

def retrain_model(**kwargs):
    import mlflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.start_run()
    train_model(NEW_DATA_PATH)
    mlflow.end_run()

def check_model_exists():
    import mlflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
    if len(runs) > 0:
        return 'detect_data_drift'
    return 'train_initial_model'

def train_initial_model():
    train_model('')
    return "detect_data_drift"

# Define DAG
with DAG(
    "data_drift_pipeline",
    default_args=default_args,
    description="A pipeline to detect data drift and retrain model",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    
    check_model = PythonOperator(
        task_id="check_model_exists",
        python_callable=check_model_exists,
    )

    train_initial = PythonOperator(
        task_id="train_initial_model",
        python_callable=train_initial_model,
    )
    
    detect_drift = PythonOperator(
        task_id="detect_data_drift",
        python_callable=detect_data_drift,
    )
    
    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model,
    )
    
    end_pipeline = PythonOperator(
        task_id="end_pipeline",
        python_callable=lambda: print("No drift detected"),
    )
    
    check_model >> [train_initial, detect_drift]
    train_initial >> detect_drift
    detect_drift >> [retrain, end_pipeline]