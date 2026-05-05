# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Inferência: pyfunc e Spark UDF
# MAGIC
# MAGIC Este notebook demonstra **dois modos de consumir** o modelo registrado
# MAGIC no Unity Catalog (notebook 04):
# MAGIC
# MAGIC 1. **Função Python** via `mlflow.pyfunc.load_model(...)` — ideal para
# MAGIC    inferência **single-record**, volumes pequenos, lógica custom no
# MAGIC    notebook ou em uma aplicação que carrega o modelo localmente.
# MAGIC 2. **Spark UDF** via `mlflow.pyfunc.spark_udf(...)` — ideal para
# MAGIC    **batch scoring** sobre tabelas grandes. O `spark_udf` distribui o
# MAGIC    modelo nos executors e paraleliza a predição. O mesmo código escala
# MAGIC    de mil para dezenas de milhões de linhas sem mudança.
# MAGIC
# MAGIC Em ambos os modos resolvemos o modelo pelo alias `Champion`
# MAGIC (`models:/<modelo>@Champion`), de forma que retreinos que movem o
# MAGIC alias passam a ser refletidos automaticamente da próxima vez que o
# MAGIC notebook rodar — sem hardcode de versão.
# MAGIC
# MAGIC ## Notebook anterior
# MAGIC
# MAGIC `04_model_registry/04_model_registry`.
# MAGIC
# MAGIC ## Próximo notebook
# MAGIC
# MAGIC `06_model_serving/06_model_serving`.

# COMMAND ----------

# MAGIC %run ../config/demo_config

# COMMAND ----------
# DBTITLE 1,Carregar configuração

get_widgets(dbutils, spark)
config = resolve_config(dbutils, spark)

print(f"Modelo        : {config.model_full_name}")
print(f"Alias         : {ALIAS_CHAMPION}")
print(f"Tabela features: {config.table(TABLE_CUSTOMER_FEATURES)}")
print(f"Coluna label   : {LABEL_COLUMN}")

# COMMAND ----------
# DBTITLE 1,Configurar MLflow para Unity Catalog

import mlflow

mlflow.set_registry_uri("databricks-uc")
print("Registry URI:", mlflow.get_registry_uri())

# URI canônica via alias — o endpoint resolve para a versão atual.
model_uri = f"models:/{config.model_full_name}@{ALIAS_CHAMPION}"
print("model_uri    :", model_uri)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Modo 1 — `mlflow.pyfunc.load_model`
# MAGIC
# MAGIC Carrega o modelo localmente no driver como um objeto Python
# MAGIC (`PyFuncModel`). Use para:
# MAGIC
# MAGIC - Predições pontuais (uma ou poucas linhas).
# MAGIC - Validação manual no notebook.
# MAGIC - Aplicações Python custom (jobs simples, scripts, APIs próprias).
# MAGIC
# MAGIC Para volumes grandes, prefira o **modo 2** (Spark UDF), que
# MAGIC paraleliza no cluster.

# COMMAND ----------
# DBTITLE 1,Modo 1 — carregar modelo e prever sobre uma amostra Pandas

model = mlflow.pyfunc.load_model(model_uri)
print("Modelo carregado:", type(model).__name__)

# Carregamos uma pequena amostra Pandas direto da feature table.
# Dropamos colunas que **não eram features** durante o treino: a label
# `will_reactivate` e o identificador `user_id`.
sample_pdf = (
    spark.table(config.table(TABLE_CUSTOMER_FEATURES))
    .limit(10)
    .toPandas()
)
sample_features = sample_pdf.drop(
    columns=[LABEL_COLUMN, "user_id"], errors="ignore"
)

print(f"Shape da amostra de features: {sample_features.shape}")
print("Colunas:", list(sample_features.columns))

preds = model.predict(sample_features)
print("\nPredições (modo pyfunc / single-process):")
print(preds)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Modo 2 — `mlflow.pyfunc.spark_udf` (batch distribuído)
# MAGIC
# MAGIC `spark_udf` registra o modelo como **UDF Spark**. Cada partição do
# MAGIC DataFrame é avaliada em paralelo nos executors do cluster, então o
# MAGIC mesmo código escala para milhões de linhas sem alteração.
# MAGIC
# MAGIC O `result_type` deve casar com o tipo de saída do modelo. Para
# MAGIC classificadores binários que retornam a classe predita, `"double"`
# MAGIC funciona bem (0.0 / 1.0). Para probabilidades, ajuste conforme o
# MAGIC modelo.

# COMMAND ----------
# DBTITLE 1,Modo 2 — registrar Spark UDF e fazer batch scoring

predict_udf = mlflow.pyfunc.spark_udf(
    spark, model_uri=model_uri, result_type="double"
)

df = spark.table(config.table(TABLE_CUSTOMER_FEATURES))

# Mesma lógica do Modo 1: features = todas as colunas exceto user_id e label.
feature_cols = [
    c for c in df.columns if c not in ("user_id", LABEL_COLUMN)
]
print(f"{len(feature_cols)} colunas de feature serão passadas ao UDF.")

scored = df.withColumn(
    "prediction", predict_udf(*[df[c] for c in feature_cols])
)

display(
    scored.select("user_id", LABEL_COLUMN, "prediction").limit(20)
)

# COMMAND ----------
# DBTITLE 1,Salvar batch scoring como tabela Delta (padrão batch inference)

# Persistir o resultado é o padrão para casos de uso "score nightly":
# o batch grava `customer_features_scored`, e consumidores downstream
# (BI, jobs de e-mail, etc.) leem essa tabela.
scored_table = config.table("customer_features_scored")
print(f"Salvando scoring em: {scored_table}")

(
    scored
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(scored_table)
)

n = spark.table(scored_table).count()
print(f"OK: {n:,} linhas escritas em {scored_table}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Inferência batch funcionando
# MAGIC
# MAGIC O que cobrimos:
# MAGIC
# MAGIC - **Modo pyfunc**: carregar e usar o modelo como função Python local.
# MAGIC - **Modo Spark UDF**: paralelizar o scoring sobre toda a feature
# MAGIC   table e persistir em `customer_features_scored` (tabela Delta no
# MAGIC   schema do participante).
# MAGIC
# MAGIC **Próximo passo**: rodar `06_model_serving/06_model_serving` para
# MAGIC servir o mesmo modelo via REST com **Mosaic AI Model Serving** e
# MAGIC habilitar **Inference Tables**.
