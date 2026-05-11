# Databricks notebook source
# MAGIC %md
# MAGIC # 03b - AutoML pelo SDK Python (programático)
# MAGIC
# MAGIC Este notebook é o **equivalente programático** ao `03a_automl_ui`. Em
# MAGIC vez de clicar pela UI, chamamos o SDK Python do **Databricks AutoML**
# MAGIC (também conhecido como Mosaic AutoML).
# MAGIC
# MAGIC Por que usar o SDK em vez da UI:
# MAGIC
# MAGIC - **Reprodutibilidade**: o notebook fica versionado no Git.
# MAGIC - **CI/CD**: é trivial agendar como Job ou parametrizar via widgets.
# MAGIC - **Integração**: o `AutoMLSummary` retornado expõe `experiment`,
# MAGIC   `best_trial.mlflow_run_id` e `best_trial.model_path`, prontos para o
# MAGIC   próximo notebook (`04_model_registry`).
# MAGIC
# MAGIC A API pode evoluir entre versões do runtime. Esta demo é validada para
# MAGIC **DBR 17.3 LTS ML**. A documentação oficial está em
# MAGIC `Machine Learning → AutoML → Python API reference` na ajuda do
# MAGIC workspace.
# MAGIC
# MAGIC **Notebook anterior**: `02_eda/02_eda` (precisa ter sido executado
# MAGIC para que `customer_features` exista).
# MAGIC
# MAGIC **Próximo notebook**: `04_model_registry/04_model_registry`.

# COMMAND ----------
# MAGIC %md
# MAGIC > **AVISO — disponibilidade do AutoML nas próximas versões do runtime**
# MAGIC >
# MAGIC > Esta demo é validada em **DBR 17.3 LTS ML**, onde o AutoML vem
# MAGIC > pré-instalado: o `import` `from databricks import automl` funciona
# MAGIC > direto, sem instalação.
# MAGIC >
# MAGIC > A partir do **DBR 18.0 ML**, o AutoML **deixa de ser uma biblioteca
# MAGIC > built-in** do runtime. Para continuar usando este notebook em
# MAGIC > DBR 18.0 ML+ será necessário instalar o pacote
# MAGIC > **`databricks-automl-runtime`** do PyPI antes do `import`:
# MAGIC >
# MAGIC > ```python
# MAGIC > %pip install databricks-automl-runtime
# MAGIC > dbutils.library.restartPython()
# MAGIC > ```
# MAGIC >
# MAGIC > Em DBR 17.3 LTS ML, **não faça nada** — o pacote já vem incluso.

# COMMAND ----------

# MAGIC %run ../config/demo_config

# COMMAND ----------
# DBTITLE 1,Carregar configuração e widget extra

get_widgets(dbutils, spark)

# Widget extra específico deste notebook: orçamento de tempo do AutoML.
dbutils.widgets.text(
    "timeout_minutes", "15", "AutoML timeout (minutos)"
)

config = resolve_config(dbutils, spark)
timeout_minutes = int(dbutils.widgets.get("timeout_minutes"))

print(f"Schema alvo        : {config.schema_full}")
print(f"Tabela de features : {config.table(TABLE_CUSTOMER_FEATURES)}")
print(f"Label              : {LABEL_COLUMN}")
print(f"Timeout (minutos)  : {timeout_minutes}")

# COMMAND ----------
# DBTITLE 1,Selecionar catálogo e schema

spark.sql(f"USE CATALOG {config.catalog}")
spark.sql(f"USE SCHEMA {config.schema}")

# COMMAND ----------
# DBTITLE 1,Pré-requisitos do runtime
# MAGIC %md
# MAGIC - Cluster **clássico single-user** com **DBR 17.3 LTS ML**. AutoML
# MAGIC   **não roda em compute serverless** nem em clusters shared.
# MAGIC - Permissões UC para ler `customer_features` e gravar experimentos
# MAGIC   no workspace.

# COMMAND ----------
# DBTITLE 1,Carregar a feature table

df = spark.table(config.table(TABLE_CUSTOMER_FEATURES))
n_rows = df.count()
print(f"Linhas carregadas: {n_rows:,}")

if n_rows == 0:
    raise RuntimeError(
        "customer_features está vazia. Rode antes o notebook 02_eda/02_eda "
        "para construir a feature table."
    )

# COMMAND ----------
# DBTITLE 1,Diretório do experimento (por usuário)

# Usamos o e-mail do `current_user()` para criar o diretório do experimento.
# Em DBR ML, `dbutils.notebook.entry_point.getDbutils().notebook().getContext()`
# expõe o contexto. Forma robusta via Spark:
user_email = (
    spark.sql("SELECT current_user() AS u").first()["u"]
)
experiment_dir = f"/Users/{user_email}/automl_reactivation_{config.user_short}"
print(f"Experiment dir: {experiment_dir}")

# COMMAND ----------
# DBTITLE 1,Rodar AutoML (classificação binária)

from databricks import automl

# API do Databricks AutoML em DBR 17.3 LTS ML.
# - dataset       : Spark / pandas / pyspark.pandas DataFrame (ou nome de tabela UC).
# - target_col    : coluna de label.
# - exclude_cols  : colunas que devem ser ignoradas como features (PKs, IDs, etc.).
# - primary_metric: métrica primária (f1, log_loss, precision, recall, roc_auc, accuracy).
# - timeout_minutes: orçamento de busca.
# - experiment_dir: pasta do workspace onde o experimento MLflow é criado.
summary = automl.classify(
    dataset=df,
    target_col=LABEL_COLUMN,
    exclude_cols=["user_id"],
    primary_metric="f1",
    timeout_minutes=timeout_minutes,
    experiment_dir=experiment_dir,
)

# COMMAND ----------
# DBTITLE 1,Resumo do best trial

best = summary.best_trial

print("=" * 72)
print("AutoML — best trial")
print("=" * 72)
print(f"experiment_id          : {summary.experiment.experiment_id}")
print(f"experiment_name        : {summary.experiment.name}")
print(f"best mlflow_run_id     : {best.mlflow_run_id}")
print(f"best model_path        : {best.model_path}")
print(f"primary metric score   : {best.evaluation_metric_score}")
print()
print("Métricas do best trial:")
for k, v in sorted(best.metrics.items()):
    print(f"  {k:<40s}  {v}")

# COMMAND ----------
# DBTITLE 1,Repassar o run_id ao próximo notebook

best_run_id = best.mlflow_run_id

# Quando este notebook roda como Job, o próximo task pode ler o run_id via
# task values (sem precisar copiar e colar manualmente).
try:
    dbutils.jobs.taskValues.set(key="best_run_id", value=best_run_id)
    print(f"taskValues['best_run_id'] = {best_run_id}")
except Exception as e:
    # Em execução interativa, taskValues pode não estar disponível — não é erro.
    print(f"(taskValues não disponível neste contexto: {e})")

print()
print("=" * 72)
print("ATENÇÃO — copie o run_id abaixo para o widget 'run_id' do notebook")
print("04_model_registry/04_model_registry caso esteja rodando manualmente:")
print("=" * 72)
print(f"  {best_run_id}")
print("=" * 72)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Modelo treinado via SDK
# MAGIC
# MAGIC O AutoML terminou, registrou todos os trials no MLflow e o
# MAGIC `best_trial.mlflow_run_id` está pronto para ser usado.
# MAGIC
# MAGIC **Próximo passo**: rodar `04_model_registry/04_model_registry`,
# MAGIC informando o `run_id` impresso acima no widget correspondente. Ele
# MAGIC vai registrar o modelo em Unity Catalog (`config.model_full_name`)
# MAGIC e atribuir o alias `Champion`.
