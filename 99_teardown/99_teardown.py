# Databricks notebook source
# MAGIC %md
# MAGIC # 99 - Teardown da demo
# MAGIC
# MAGIC **Rode este notebook ao final da demo — é obrigatório.**
# MAGIC
# MAGIC O endpoint de Mosaic AI Model Serving (criado no notebook 06)
# MAGIC continua provisionado mesmo sem tráfego — `scale_to_zero=true`
# MAGIC reduz, mas **não zera**, o custo de manter o endpoint ativo: o
# MAGIC workspace mantém metadata e reservas mínimas associadas. A maneira
# MAGIC correta de não pagar mais é **deletar o endpoint**.
# MAGIC
# MAGIC Este notebook também oferece (opcional) o drop do seu schema,
# MAGIC para limpar tabelas e o modelo registrado em UC.
# MAGIC
# MAGIC ## Notebook anterior
# MAGIC
# MAGIC `07_monitoring/07_monitoring`.

# COMMAND ----------

# MAGIC %run ../config/demo_config

# COMMAND ----------
# DBTITLE 1,Carregar configuração e widget extra

get_widgets(dbutils, spark)

# Widget extra: confirmação explícita para dropar o schema. Default false
# por segurança — você precisa marcar `true` deliberadamente.
dbutils.widgets.dropdown(
    "confirm_drop_schema",
    "false",
    ["false", "true"],
    "Dropar seu schema? (CASCADE)",
)

config = resolve_config(dbutils, spark)
confirm_drop = dbutils.widgets.get("confirm_drop_schema").lower() == "true"

print(f"Endpoint a deletar : {config.endpoint_name}")
print(f"Schema da demo     : {config.schema_full}")
print(f"Drop do schema?    : {confirm_drop}")

# COMMAND ----------
# DBTITLE 1,Deletar o endpoint de Model Serving

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

try:
    w.serving_endpoints.delete(name=config.endpoint_name)
    print(f"OK: endpoint deletado -> {config.endpoint_name}")
except Exception as e:
    # Idempotência: se já foi deletado, ou nunca existiu, não falhar.
    print(
        f"Endpoint '{config.endpoint_name}' não encontrado ou já deletado.\n"
        f"  ({type(e).__name__}: {e})"
    )

# COMMAND ----------
# DBTITLE 1,Drop do schema (condicional)

if confirm_drop:
    sql = f"DROP SCHEMA IF EXISTS {config.schema_full} CASCADE"
    print(f"Executando: {sql}")
    spark.sql(sql)
    print(f"OK: schema dropado -> {config.schema_full}")
else:
    print(
        f"Schema {config.schema_full} preservado.\n"
        "  Para dropar, marque o widget 'confirm_drop_schema' = true e "
        "rode este notebook novamente."
    )

# COMMAND ----------
# DBTITLE 1,Sanity check — endpoints e tabelas remanescentes

# Lista endpoints servindo o modelo desta demo. Se o delete acima
# funcionou, este conjunto deve estar vazio.
print("Endpoints servindo o modelo da demo:")
print("-" * 60)
restantes = []
try:
    for ep in w.serving_endpoints.list():
        config_obj = getattr(ep, "config", None)
        served_entities = (
            getattr(config_obj, "served_entities", None) if config_obj else None
        ) or []
        for se in served_entities:
            entity_name = getattr(se, "entity_name", None)
            if entity_name == config.model_full_name:
                restantes.append(ep.name)
                break
    if restantes:
        for name in restantes:
            print(f"  - {name}")
    else:
        print("  (nenhum)")
except Exception as e:
    print(f"  (não foi possível listar endpoints: {e})")

# Lista tabelas do schema (se preservado).
print()
print(f"Tabelas em {config.schema_full}:")
print("-" * 60)
if confirm_drop:
    print("  (schema dropado, lista vazia)")
else:
    try:
        tables = spark.sql(f"SHOW TABLES IN {config.schema_full}").collect()
        if not tables:
            print("  (nenhuma)")
        for row in tables:
            print(f"  - {row['tableName']}")
    except Exception as e:
        print(f"  (não foi possível listar tabelas: {e})")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Demo finalizada
# MAGIC
# MAGIC - Endpoint deletado (ou já não existia).
# MAGIC - Schema preservado por padrão; opcionalmente dropado se o widget
# MAGIC   `confirm_drop_schema` foi marcado como `true`.
# MAGIC
# MAGIC Obrigado por percorrer a demo de AutoML + MLflow + Model Serving +
# MAGIC Inference Tables. Para rodar de novo do zero, basta começar pelo
# MAGIC notebook `00_setup/00_setup`.
