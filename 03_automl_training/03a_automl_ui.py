# Databricks notebook source
# MAGIC %md
# MAGIC # 03a - AutoML pela UI (walkthrough)
# MAGIC
# MAGIC Este notebook **não executa** o AutoML — ele te guia para rodar
# MAGIC o AutoML pela **UI do workspace** sobre a tabela
# MAGIC `customer_features` criada em `02_eda`.
# MAGIC
# MAGIC O equivalente programático está em `03b_automl_sdk.py`. Os dois são
# MAGIC **intercambiáveis**: rode apenas um.
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
# MAGIC > pré-instalado tanto na UI quanto via biblioteca `databricks.automl`.
# MAGIC >
# MAGIC > A partir do **DBR 18.0 ML**, o AutoML **deixa de ser uma biblioteca
# MAGIC > built-in** do runtime. Para continuar usando o fluxo deste notebook
# MAGIC > em DBR 18.0 ML+ será necessário instalar manualmente o pacote do PyPI
# MAGIC > **`databricks-automl-runtime`** no cluster (ou notebook):
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
# DBTITLE 1,Carregar configuração

get_widgets(dbutils, spark)
config = resolve_config(dbutils, spark)

# COMMAND ----------
# DBTITLE 1,Imprimir os nomes que você vai usar na UI

print("Use estes nomes ao preencher o formulário do AutoML na UI:\n")
print(f"  Dataset (UC table)   : {config.table(TABLE_CUSTOMER_FEATURES)}")
print(f"  Prediction target    : {LABEL_COLUMN}")
print(f"  Sugestão experiment  : automl_reactivation_{config.user_short}")
print(f"  Excluir da feature   : user_id  (é PK, não tem sinal)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Pré-requisitos
# MAGIC
# MAGIC - **Cluster clássico** (modo *single-user*) com runtime
# MAGIC   **DBR 17.3 LTS ML**. O AutoML **não roda em compute serverless**
# MAGIC   nem em clusters *shared / no-isolation*.
# MAGIC - Permissões Unity Catalog: `USE CATALOG`, `USE SCHEMA`, `SELECT`
# MAGIC   sobre a tabela `customer_features`.
# MAGIC - A tabela `customer_features` precisa existir — ou seja, o notebook
# MAGIC   `02_eda/02_eda` já foi executado.
# MAGIC
# MAGIC No Databricks atual, o produto se chama **Mosaic AutoML**, mas o fluxo
# MAGIC pela UI continua sendo iniciado em **Experiments → Create AutoML
# MAGIC Experiment**.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Passo 1 — Abrir o formulário de AutoML
# MAGIC
# MAGIC Na barra lateral do workspace, vá em uma das opções abaixo (qualquer
# MAGIC uma chega ao mesmo formulário):
# MAGIC
# MAGIC 1. **Experiments → Create AutoML Experiment**, **ou**
# MAGIC 2. **Machine Learning → Experiments → Create AutoML Experiment**.
# MAGIC
# MAGIC Você verá um formulário com seções: *Compute*, *ML problem type*,
# MAGIC *Dataset*, *Prediction target* e *Advanced configuration*.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Passo 2 — Compute
# MAGIC
# MAGIC No campo *Compute*, selecione um **cluster clássico em modo
# MAGIC single-user** rodando **DBR 17.3 LTS ML**.
# MAGIC
# MAGIC - Single-user é obrigatório para AutoML acessar Unity Catalog.
# MAGIC - **Não use** clusters *shared*, *no-isolation* ou *serverless* —
# MAGIC   o AutoML clássico precisa de um cluster persistente.
# MAGIC - Se não houver um cluster compatível, crie um novo com runtime
# MAGIC   `ML 17.3 LTS` antes de continuar.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Passo 3 — ML problem type
# MAGIC
# MAGIC Selecione **`Classification`**.
# MAGIC
# MAGIC O label `will_reactivate` é binário (0/1), então é classificação
# MAGIC binária. Não selecione *Regression* nem *Forecasting*.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Passo 4 — Dataset
# MAGIC
# MAGIC Em *Dataset*, escolha **Unity Catalog table** e digite/selecione a
# MAGIC tabela. O nome exato — copie o que foi impresso na primeira célula:

# COMMAND ----------

print(config.table(TABLE_CUSTOMER_FEATURES))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Passo 5 — Prediction target
# MAGIC
# MAGIC Em *Prediction target*, selecione a coluna alvo. O nome exato:

# COMMAND ----------

print(LABEL_COLUMN)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Passo 6 — Experiment name (opcional)
# MAGIC
# MAGIC Você pode deixar o nome padrão ou usar a sugestão abaixo. Um nome com
# MAGIC seu user_short facilita encontrar o experimento depois.

# COMMAND ----------

print(f"automl_reactivation_{config.user_short}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Passo 7 — Advanced configuration
# MAGIC
# MAGIC Abra a seção *Advanced configuration* e ajuste:
# MAGIC
# MAGIC - **Timeout (minutes)**: `10` a `15` minutos para a demo. Um budget
# MAGIC   maior produz modelos melhores mas demora mais.
# MAGIC - **Evaluation metric**: deixe o padrão (**F1** ou **ROC AUC**) — são
# MAGIC   adequadas para classificação binária com classes desbalanceadas.
# MAGIC   Evite *accuracy* aqui.
# MAGIC - **Excluded columns**: adicione **`user_id`**. É a chave primária da
# MAGIC   feature table e não carrega sinal preditivo (e cardinalidade muito
# MAGIC   alta degradaria os modelos).
# MAGIC - **Positive label** (opcional): `1`.
# MAGIC
# MAGIC Você pode deixar os demais campos (data dir, frameworks excluídos,
# MAGIC imputers) com os valores padrão.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Passo 8 — Start AutoML
# MAGIC
# MAGIC Clique em **Start AutoML**. O que esperar enquanto roda:
# MAGIC
# MAGIC 1. O AutoML faz **data exploration** automática e gera um notebook de
# MAGIC    *Data Exploration* (estatísticas, missing, correlações).
# MAGIC 2. Em paralelo, treina **vários trials** com diferentes algoritmos
# MAGIC    (`scikit-learn`, `xgboost`, `lightgbm`) e hiperparâmetros.
# MAGIC 3. Cada trial é registrado como um run MLflow no experimento criado.
# MAGIC 4. Ao final, você terá:
# MAGIC    - O **best trial** marcado na UI do experimento.
# MAGIC    - Um **notebook de código gerado** para o melhor trial — totalmente
# MAGIC      reproduzível e editável.
# MAGIC    - O **Data Exploration notebook** com a EDA automática.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Passo 9 — Localizar o resultado
# MAGIC
# MAGIC Depois que o AutoML terminar:
# MAGIC
# MAGIC 1. Abra o **experimento** (link no topo da página de progresso).
# MAGIC 2. Ordene os runs pela métrica primária (F1 ou ROC AUC) — o **best
# MAGIC    run** já vem marcado.
# MAGIC 3. Clique no best run e **copie o `Run ID`** (canto superior).
# MAGIC 4. Opcional: abra o **Source notebook** gerado para ver o código
# MAGIC    exato do modelo vencedor.
# MAGIC
# MAGIC Anote esse `Run ID` — você vai colar no widget `run_id` do notebook
# MAGIC `04_model_registry`, que faz o registro do modelo em UC com o alias
# MAGIC `Champion`.

# COMMAND ----------
# DBTITLE 1,Listar experimentos AutoML do seu usuário

import mlflow

experiments = mlflow.search_experiments(
    filter_string=f"name LIKE '%{config.user_short}%'"
)

if not experiments:
    print(
        "Nenhum experimento encontrado contendo "
        f"'{config.user_short}' no nome.\n"
        "Se você acabou de iniciar o AutoML pela UI, espere alguns segundos\n"
        "e rode esta célula de novo."
    )
else:
    print(f"Experimentos contendo '{config.user_short}':\n")
    print(f"  {'experiment_id':<20s}  name")
    for e in experiments:
        print(f"  {e.experiment_id:<20s}  {e.name}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Modelo treinado pela UI
# MAGIC
# MAGIC Com o `Run ID` do best trial em mãos, você pode prosseguir.
# MAGIC
# MAGIC **Próximo passo**: rodar `04_model_registry/04_model_registry` para
# MAGIC registrar o modelo em Unity Catalog com o alias `Champion`.
