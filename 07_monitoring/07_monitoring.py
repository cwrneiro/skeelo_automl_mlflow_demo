# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Monitoring via Inference Tables
# MAGIC
# MAGIC Este notebook consome as **Inference Tables** geradas pelo endpoint
# MAGIC (notebook 06) para entender uso, latência, distribuição de
# MAGIC previsões e um esboço de **drift** comparando a distribuição de
# MAGIC features no treino vs. no tráfego de produção.
# MAGIC
# MAGIC ## Esquema da Inference Table
# MAGIC
# MAGIC Quando `auto_capture_config` está habilitada, o Databricks cria
# MAGIC automaticamente a tabela
# MAGIC `<catalog>.<schema>.<table_name_prefix>_payload` (sufixo `_payload`
# MAGIC confirmado na doc oficial). Cada linha é um request capturado,
# MAGIC contendo:
# MAGIC
# MAGIC | coluna | tipo | conteúdo |
# MAGIC |---|---|---|
# MAGIC | `databricks_request_id` | STRING | id do request gerado pelo serving |
# MAGIC | `client_request_id` | STRING | id opcional fornecido pelo cliente |
# MAGIC | `date` | DATE | data UTC em que o request foi recebido |
# MAGIC | `timestamp_ms` | LONG | epoch ms |
# MAGIC | `status_code` | INT | HTTP status code da resposta |
# MAGIC | `sampling_fraction` | DOUBLE | fração de sampling (0..1) |
# MAGIC | `execution_time_ms` | LONG | latência do scoring |
# MAGIC | `request` | STRING | payload JSON do request |
# MAGIC | `response` | STRING | payload JSON da resposta |
# MAGIC | `request_metadata` | MAP<STRING,STRING> | endpoint/model/version |
# MAGIC
# MAGIC ## Observação importante
# MAGIC
# MAGIC Esta análise é um **esboço pedagógico**. Em produção real você
# MAGIC usaria **Lakehouse Monitoring** sobre a Inference Table, que dá
# MAGIC drift, data quality, alertas e dashboard automaticamente.
# MAGIC
# MAGIC ## Notebook anterior
# MAGIC
# MAGIC `06_model_serving/06_model_serving`.
# MAGIC
# MAGIC ## Próximo notebook
# MAGIC
# MAGIC `99_teardown/99_teardown`.

# COMMAND ----------

# MAGIC %run ../config/demo_config

# COMMAND ----------
# DBTITLE 1,Carregar configuração

get_widgets(dbutils, spark)
config = resolve_config(dbutils, spark)

# Sufixo `_payload` é o nome canônico documentado pela Databricks para
# a tabela gerada pelo auto_capture_config. Ver:
#   https://docs.databricks.com/aws/en/machine-learning/model-serving/inference-tables.html
inference_table = config.table(f"{INFERENCE_TABLE_PREFIX}_payload")
features_table = config.table(TABLE_CUSTOMER_FEATURES)

print(f"Inference table : {inference_table}")
print(f"Features table  : {features_table}")

# COMMAND ----------
# DBTITLE 1,Sanity check — a tabela existe?

if not spark.catalog.tableExists(inference_table):
    raise RuntimeError(
        f"Tabela {inference_table} ainda não existe.\n"
        "  Inference Tables têm latência de ~5 a 10 minutos após a primeira\n"
        "  chamada ao endpoint. Possíveis causas:\n"
        "    - Você acabou de rodar 06_model_serving e o flush ainda não\n"
        "      ocorreu. Espere alguns minutos.\n"
        "    - O endpoint ainda não recebeu nenhum request. Volte ao\n"
        "      notebook 06 e rode a célula 'Chamada REST' algumas vezes.\n"
        "    - O auto_capture_config não foi habilitado. Verifique no\n"
        "      notebook 06 que `enabled=True` foi passado."
    )

n_rows = spark.table(inference_table).count()
print(f"OK: {inference_table} existe com {n_rows:,} linhas.")
if n_rows == 0:
    print(
        "AVISO: tabela existe, mas está vazia. Faça mais chamadas REST ao "
        "endpoint (notebook 06) e re-execute as próximas células."
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## Uso do endpoint
# MAGIC
# MAGIC Volume de requests por dia, latência (p50, p95) e distribuição de
# MAGIC `status_code`. Consultas em SQL para reaproveitamento direto em
# MAGIC dashboards.

# COMMAND ----------
# DBTITLE 1,Volume de requests por dia

display(spark.sql(f"""
    SELECT
        date,
        COUNT(*) AS requests,
        SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END) AS ok,
        SUM(CASE WHEN status_code <> 200 THEN 1 ELSE 0 END) AS errors
    FROM {inference_table}
    GROUP BY date
    ORDER BY date
"""))

# COMMAND ----------
# DBTITLE 1,Latência (p50, p95) e contagem de status_code

display(spark.sql(f"""
    SELECT
        status_code,
        COUNT(*) AS n,
        percentile_approx(execution_time_ms, 0.50) AS p50_ms,
        percentile_approx(execution_time_ms, 0.95) AS p95_ms,
        MAX(execution_time_ms) AS max_ms
    FROM {inference_table}
    GROUP BY status_code
    ORDER BY status_code
"""))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Distribuição das previsões
# MAGIC
# MAGIC As predições estão **dentro do JSON** da coluna `response`. Para
# MAGIC modelos pyfunc o formato canônico é `{"predictions": [v1, v2, ...]}`.
# MAGIC Usamos `from_json` com um schema simples e `explode` para extrair
# MAGIC cada predição como uma linha.

# COMMAND ----------
# DBTITLE 1,Extrair previsões do response e plotar histograma

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, ArrayType, DoubleType

response_schema = StructType([
    StructField("predictions", ArrayType(DoubleType()), True),
])

predictions_df = (
    spark.table(inference_table)
    .where("status_code = 200")
    .withColumn("resp", F.from_json("response", response_schema))
    .select("date", F.explode("resp.predictions").alias("prediction"))
)

print("Estatísticas das previsões capturadas:")
predictions_df.selectExpr(
    "COUNT(*) AS n",
    "AVG(prediction) AS mean",
    "MIN(prediction) AS min",
    "MAX(prediction) AS max",
    "percentile_approx(prediction, 0.5) AS p50",
).show()

# Histograma simples por bucket arredondado.
display(
    predictions_df
    .groupBy(F.round("prediction", 2).alias("prediction_bucket"))
    .count()
    .orderBy("prediction_bucket")
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Drift básico de uma feature
# MAGIC
# MAGIC Comparamos a distribuição da feature `events_last_90d` (escolhida
# MAGIC porque é numérica, intuitiva e ligada à reativação) entre:
# MAGIC
# MAGIC - **Treino**: tabela `customer_features` (notebook 02).
# MAGIC - **Produção**: parsing do payload `request` da Inference Table.
# MAGIC
# MAGIC Diferenças grandes em média / percentis indicam que o tráfego que
# MAGIC chega ao endpoint não tem mais a mesma distribuição do que o treino
# MAGIC viu — sinal de drift.

# COMMAND ----------
# DBTITLE 1,Estatísticas da feature no treino

drift_feature = "events_last_90d"

print(f"Feature analisada: {drift_feature}")

train_df = spark.table(features_table)
if drift_feature not in train_df.columns:
    print(
        f"AVISO: a feature '{drift_feature}' não está em {features_table}. "
        "Ajuste a variável `drift_feature` para uma coluna existente."
    )
else:
    print("\nDistribuição no treino:")
    train_df.selectExpr(
        f"COUNT({drift_feature}) AS n",
        f"AVG({drift_feature}) AS mean",
        f"STDDEV({drift_feature}) AS stddev",
        f"percentile_approx({drift_feature}, 0.5) AS p50",
        f"percentile_approx({drift_feature}, 0.95) AS p95",
    ).show()

# COMMAND ----------
# DBTITLE 1,Estatísticas da feature na produção (parsing do request)

# O request enviado pelo cliente segue o formato `dataframe_split`:
#   {"dataframe_split": {"columns": [...], "data": [[...], [...]]}}
# Para extrair de forma robusta, usamos `from_json` com um schema mínimo
# e depois fazemos `arrays_zip` entre `columns` e `data` para mapear cada
# valor à sua coluna.

from pyspark.sql.types import StringType, IntegerType, LongType

request_schema = StructType([
    StructField(
        "dataframe_split",
        StructType([
            StructField("columns", ArrayType(StringType()), True),
            StructField("data", ArrayType(ArrayType(StringType())), True),
        ]),
        True,
    )
])

prod_raw = (
    spark.table(inference_table)
    .where("status_code = 200")
    .withColumn("req", F.from_json("request", request_schema))
    .select("date", "req.dataframe_split.columns", "req.dataframe_split.data")
)

# Para cada linha do request (cada array em `data`), encontramos o índice
# de `drift_feature` em `columns` e extraímos o valor correspondente.
prod_values = (
    prod_raw
    .withColumn("col_idx", F.array_position("columns", F.lit(drift_feature)) - 1)
    .where("col_idx >= 0")
    .withColumn("row", F.explode("data"))
    .selectExpr(
        "date",
        f"CAST(row[col_idx] AS DOUBLE) AS {drift_feature}",
    )
    .where(F.col(drift_feature).isNotNull())
)

n_prod = prod_values.count()
print(f"Linhas extraídas do tráfego de produção: {n_prod}")

if n_prod == 0:
    print(
        "Sem dados de produção para essa feature. Possíveis causas:\n"
        "  - O request foi enviado em outro formato (não `dataframe_split`).\n"
        "  - A feature drift_feature não estava em `columns` do request.\n"
        "  - Ainda não houve flush da Inference Table — espere alguns minutos."
    )
else:
    print("Distribuição na produção:")
    prod_values.selectExpr(
        f"COUNT({drift_feature}) AS n",
        f"AVG({drift_feature}) AS mean",
        f"STDDEV({drift_feature}) AS stddev",
        f"percentile_approx({drift_feature}, 0.5) AS p50",
        f"percentile_approx({drift_feature}, 0.95) AS p95",
    ).show()

    print(
        "\nLeitura: comparar `mean`, `stddev`, `p50`, `p95` entre treino e\n"
        "produção. Diferenças grandes (ex: > 20%) sugerem drift e merecem\n"
        "investigação — possivelmente retreino com dados mais recentes."
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## Monitoring concluído
# MAGIC
# MAGIC O que cobrimos a partir das Inference Tables:
# MAGIC
# MAGIC - Volume de requests por dia.
# MAGIC - Latência p50/p95 e distribuição de `status_code`.
# MAGIC - Distribuição das previsões capturadas.
# MAGIC - Esboço de drift (treino vs. produção) em uma feature numérica.
# MAGIC
# MAGIC Em produção real, recomendamos **Lakehouse Monitoring** sobre a
# MAGIC Inference Table — ele constrói o profile, calcula drift e métricas
# MAGIC de qualidade ao longo do tempo e gera dashboard sem código.
# MAGIC
# MAGIC **Próximo passo**: rodar `99_teardown/99_teardown` para deletar o
# MAGIC endpoint (custo) e, opcionalmente, dropar o schema da demo.
