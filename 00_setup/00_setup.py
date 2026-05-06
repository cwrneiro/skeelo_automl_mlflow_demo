# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Setup do ambiente
# MAGIC
# MAGIC Este é o **primeiro notebook** da demo. Seu objetivo é validar que o
# MAGIC participante tem permissões mínimas no Unity Catalog para criar o schema
# MAGIC da demo, criar tabelas Delta e usar as bibliotecas necessárias do
# MAGIC runtime DBR 17.3 LTS ML.
# MAGIC
# MAGIC **Caso de uso da demo**: predição de **reativação de clientes** em uma
# MAGIC plataforma genérica de audiobooks e ebooks, usando AutoML, MLflow,
# MAGIC Model Serving e Inference Tables.
# MAGIC
# MAGIC **Ordem de execução dos notebooks**:
# MAGIC 1. `00_setup/00_setup` (este notebook)
# MAGIC 2. `01_data_generation/01_data_generation`
# MAGIC 3. `02_eda/02_eda`
# MAGIC 4. `03_automl_training/03_automl_training`
# MAGIC 5. `04_model_registry/04_model_registry`
# MAGIC 6. `05_inference_notebook/05_inference_notebook`
# MAGIC 7. `06_model_serving/06_model_serving`
# MAGIC 8. `07_monitoring/07_monitoring`
# MAGIC 9. `99_teardown/99_teardown`
# MAGIC
# MAGIC **Pré-requisitos**:
# MAGIC - Cluster ou warehouse com runtime **DBR 17.3 LTS ML** (ou superior).
# MAGIC - Permissões `USE CATALOG` e `CREATE SCHEMA` no catálogo informado.
# MAGIC - Permissão `CREATE TABLE` no schema do participante.

# COMMAND ----------

# MAGIC %run ../config/demo_config

# COMMAND ----------
# DBTITLE 1,Carregar configuração

get_widgets(dbutils, spark)
config = resolve_config(dbutils, spark)
print("Configuração resolvida:")
print(f"  catalog       = {config.catalog}")
print(f"  schema        = {config.schema}")
print(f"  schema_full   = {config.schema_full}")
print(f"  endpoint_name = {config.endpoint_name}")
print(f"  snapshot_date = {config.snapshot_date}")
print(f"  user_short    = {config.user_short}")
print(f"  model_full    = {config.model_full_name}")

# COMMAND ----------
# DBTITLE 1,Validar permissões de catálogo e criar schema

try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.schema_full}")
    print(f"OK: schema '{config.schema_full}' disponível.")
except Exception as e:
    msg = (
        "FALHA ao criar/garantir o schema da demo.\n"
        f"  Schema alvo: {config.schema_full}\n"
        "  Você precisa das seguintes permissões no catálogo informado:\n"
        f"    - USE CATALOG no catálogo '{config.catalog}'\n"
        f"    - CREATE SCHEMA no catálogo '{config.catalog}'\n"
        "  Ajuste o widget 'catalog' para um catálogo onde você tenha esses\n"
        "  privilégios, ou peça ao administrador para concedê-los.\n"
        f"  Erro original: {e}"
    )
    raise RuntimeError(msg) from e

# COMMAND ----------
# DBTITLE 1,Selecionar catálogo e schema

spark.sql(f"USE CATALOG {config.catalog}")
spark.sql(f"USE SCHEMA {config.schema}")
print(f"USE CATALOG {config.catalog}; USE SCHEMA {config.schema};")

# COMMAND ----------
# DBTITLE 1,Smoke test de CREATE TABLE

smoke_table = config.table("_demo_smoke_test")
try:
    spark.sql(f"DROP TABLE IF EXISTS {smoke_table}")
    spark.sql(
        f"CREATE TABLE {smoke_table} (id INT, msg STRING) USING DELTA"
    )
    spark.sql(f"INSERT INTO {smoke_table} VALUES (1, 'ok')")
    rows = spark.sql(f"SELECT * FROM {smoke_table}").collect()
    if len(rows) != 1 or rows[0]["msg"] != "ok":
        raise RuntimeError(
            f"Leitura inesperada da tabela de smoke test: {rows}"
        )
    spark.sql(f"DROP TABLE {smoke_table}")
    print(f"OK: criação, leitura e drop de tabela funcionam em {config.schema_full}.")
except Exception as e:
    # Tenta limpar a tabela de smoke test mesmo em caso de falha
    try:
        spark.sql(f"DROP TABLE IF EXISTS {smoke_table}")
    except Exception:
        pass
    msg = (
        "FALHA no smoke test de CREATE TABLE.\n"
        f"  Schema alvo: {config.schema_full}\n"
        "  Você precisa de permissão CREATE TABLE no schema da demo.\n"
        "  Verifique também se o schema foi de fato criado pelo passo anterior\n"
        "  e se o catálogo aceita tabelas gerenciadas Delta.\n"
        f"  Erro original: {e}"
    )
    raise RuntimeError(msg) from e

# COMMAND ----------
# DBTITLE 1,Verificar bibliotecas do runtime

import sys
print(f"Python: {sys.version.split()[0]}")

try:
    import mlflow
    print(f"mlflow: {mlflow.__version__}")
except Exception as e:
    print(f"AVISO: não foi possível importar mlflow: {e}")

try:
    import databricks.sdk as dbsdk
    print(f"databricks-sdk: {getattr(dbsdk, '__version__', 'desconhecida')}")
except Exception as e:
    print(f"AVISO: não foi possível importar databricks.sdk: {e}")

try:
    import pandas as pd
    print(f"pandas: {pd.__version__}")
except Exception as e:
    print(f"AVISO: não foi possível importar pandas: {e}")

try:
    import numpy as np
    print(f"numpy: {np.__version__}")
except Exception as e:
    print(f"AVISO: não foi possível importar numpy: {e}")

print(
    "\nNenhuma instalação adicional é necessária. O runtime DBR 17.3 LTS ML "
    "já traz todas as dependências usadas pela demo."
)

# COMMAND ----------
# DBTITLE 1,Persistir configuração para os demais notebooks

# Grava a configuração resolvida num arquivo no Workspace do usuário. Os demais
# notebooks lerão esse arquivo como *default* dos widgets, então o participante
# não precisa repreencher catalog/schema/endpoint em cada um.
saved_path = save_config(config, spark)
print(f"Configuração persistida em: {saved_path}")
print(
    "Os próximos notebooks (01..99) já vão abrir com esses valores nos "
    "widgets. Para alterar a configuração global, edite os widgets neste "
    "notebook 00_setup e rode-o de novo."
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup concluído
# MAGIC
# MAGIC Permissões validadas, schema do participante pronto e configuração
# MAGIC persistida no Workspace.
# MAGIC
# MAGIC **Próximo passo**: rodar `01_data_generation/01_data_generation` para
# MAGIC gerar as tabelas mock da demo. Os widgets já virão preenchidos.
