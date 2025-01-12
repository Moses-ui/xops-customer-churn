FROM apache/airflow:2.10.4

ENV SPARK_VERSION 3.5.4
ENV HADOOP_VERSION 3

USER root
ARG openjdk_version="17"

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    "openjdk-${openjdk_version}-jre-headless" \
    ca-certificates-java wget tar && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget -q https://downloads.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

ENV SPARK_HOME /opt/spark
ENV PATH $SPARK_HOME/bin:$PATH

USER airflow
RUN pip install mlflow pyspark