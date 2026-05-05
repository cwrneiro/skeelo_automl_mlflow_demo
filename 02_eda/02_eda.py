# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - EDA e construção da feature table
# MAGIC
# MAGIC Neste notebook vamos:
# MAGIC
# MAGIC 1. **Explorar** as quatro tabelas brutas geradas em
# MAGIC    `01_data_generation` (`users`, `books`, `reading_events`,
# MAGIC    `subscriptions`) com queries SQL e Python+`display()`.
# MAGIC 2. **Definir o label** `will_reactivate` a partir das janelas de
# MAGIC    inatividade e horizonte de previsão.
# MAGIC 3. **Construir a feature table** `customer_features`, que será o
# MAGIC    **input do AutoML** nos notebooks `03a_automl_ui` e `03b_automl_sdk`.
# MAGIC
# MAGIC **Notebook anterior**: `01_data_generation/01_data_generation`.
# MAGIC
# MAGIC **Próximo notebook**: `03_automl_training/03a_automl_ui` (UI) ou
# MAGIC `03_automl_training/03b_automl_sdk` (programático). Os dois são
# MAGIC intercambiáveis — escolha um.

# COMMAND ----------

# MAGIC %run ../config/demo_config

# COMMAND ----------
# DBTITLE 1,Carregar configuração

get_widgets(dbutils, spark)
config = resolve_config(dbutils, spark)
print(f"Schema alvo: {config.schema_full}")
print(f"Snapshot date: {config.snapshot_date}")
print(f"Janela de inatividade: {INACTIVITY_WINDOW_DAYS} dias")
print(f"Horizonte de previsão: {PREDICTION_HORIZON_DAYS} dias")

# COMMAND ----------
# DBTITLE 1,Selecionar catálogo e schema

spark.sql(f"USE CATALOG {config.catalog}")
spark.sql(f"USE SCHEMA {config.schema}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parte 1 — Exploração das tabelas brutas
# MAGIC
# MAGIC Antes de modelar, vamos entender os dados. As próximas células alternam
# MAGIC SQL (`%sql`) e Python+`display()` para mostrar o catálogo, distribuições
# MAGIC e sazonalidade dos eventos.

# COMMAND ----------
# DBTITLE 1,Distribuição de plan_type, device_type e region em users
# MAGIC %sql
# MAGIC SELECT plan_type, device_type, region, COUNT(*) AS n
# MAGIC FROM users
# MAGIC GROUP BY plan_type, device_type, region
# MAGIC ORDER BY n DESC

# COMMAND ----------
# DBTITLE 1,Distribuição de format em books
# MAGIC %sql
# MAGIC SELECT format, COUNT(*) AS n_books, ROUND(AVG(length_minutes), 1) AS avg_minutes
# MAGIC FROM books
# MAGIC GROUP BY format

# COMMAND ----------
# DBTITLE 1,Top gêneros por volume de eventos
# MAGIC %sql
# MAGIC SELECT b.genre, COUNT(*) AS n_events
# MAGIC FROM reading_events e
# MAGIC JOIN books b ON e.book_id = b.book_id
# MAGIC GROUP BY b.genre
# MAGIC ORDER BY n_events DESC

# COMMAND ----------
# DBTITLE 1,Histograma de eventos por usuário (Pareto)

events_per_user = spark.sql(
    """
    SELECT user_id, COUNT(*) AS n_events
    FROM reading_events
    GROUP BY user_id
    """
)
display(events_per_user)

# COMMAND ----------
# DBTITLE 1,Eventos por mês (sazonalidade)
# MAGIC %sql
# MAGIC SELECT
# MAGIC   DATE_TRUNC('month', event_ts) AS month,
# MAGIC   COUNT(*) AS n_events
# MAGIC FROM reading_events
# MAGIC GROUP BY month
# MAGIC ORDER BY month

# COMMAND ----------
# DBTITLE 1,Subscriptions: % canceladas e top cancel_reason
# MAGIC %sql
# MAGIC SELECT
# MAGIC   COUNT(*) AS total_subs,
# MAGIC   SUM(CASE WHEN end_date IS NOT NULL THEN 1 ELSE 0 END) AS canceladas,
# MAGIC   ROUND(100.0 * SUM(CASE WHEN end_date IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_canceladas
# MAGIC FROM subscriptions

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT cancel_reason, COUNT(*) AS n
# MAGIC FROM subscriptions
# MAGIC WHERE cancel_reason IS NOT NULL
# MAGIC GROUP BY cancel_reason
# MAGIC ORDER BY n DESC

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parte 2 — Definição do label `will_reactivate`
# MAGIC
# MAGIC O problema de negócio é **prever reativação** de clientes que ficaram
# MAGIC inativos. A definição operacional do label é:
# MAGIC
# MAGIC - **Snapshot date** (`config.snapshot_date`): data de corte que separa o
# MAGIC   passado (features) do futuro (label).
# MAGIC - **Janela de inatividade** (`INACTIVITY_WINDOW_DAYS = 30`): um usuário
# MAGIC   é **elegível** se **não tiver nenhum evento** nos últimos 30 dias
# MAGIC   antes do snapshot. Ou seja, está inativo no corte.
# MAGIC - **Horizonte de previsão** (`PREDICTION_HORIZON_DAYS = 30`):
# MAGIC   `will_reactivate = 1` se o usuário elegível tiver **algum evento** em
# MAGIC   `[snapshot, snapshot + 30 dias)`. Caso contrário, 0.
# MAGIC
# MAGIC Apenas usuários elegíveis entram em `customer_features`. Usuários
# MAGIC ativos no corte não fazem parte do problema.
# MAGIC
# MAGIC As **features** são calculadas usando **somente eventos históricos**
# MAGIC (`event_ts < snapshot`) — para evitar vazamento (data leakage).

# COMMAND ----------
# DBTITLE 1,Imports e variáveis de janela

from datetime import timedelta
from pyspark.sql import functions as F
from pyspark.sql import Window

snapshot = config.snapshot_date
inactivity_cutoff = snapshot - timedelta(days=INACTIVITY_WINDOW_DAYS)
horizon_end = snapshot + timedelta(days=PREDICTION_HORIZON_DAYS)

print(f"Snapshot date         : {snapshot}")
print(f"Inactivity cutoff     : {inactivity_cutoff}  (eventos a partir daqui = ativo no corte)")
print(f"Horizonte (label end) : {horizon_end}")

# COMMAND ----------
# DBTITLE 1,Carregar dados

users = spark.table(config.table(TABLE_USERS))
books = spark.table(config.table(TABLE_BOOKS))
events = spark.table(config.table(TABLE_READING_EVENTS))
subs = spark.table(config.table(TABLE_SUBSCRIPTIONS))

print(f"users          : {users.count():,}")
print(f"books          : {books.count():,}")
print(f"reading_events : {events.count():,}")
print(f"subscriptions  : {subs.count():,}")

# COMMAND ----------
# DBTITLE 1,Particionar eventos: histórico vs futuro

snapshot_ts = F.lit(snapshot.isoformat()).cast("timestamp")
horizon_ts = F.lit(horizon_end.isoformat()).cast("timestamp")
inactivity_ts = F.lit(inactivity_cutoff.isoformat()).cast("timestamp")

events_hist = events.where(F.col("event_ts") < snapshot_ts)
events_future = events.where(
    (F.col("event_ts") >= snapshot_ts) & (F.col("event_ts") < horizon_ts)
)

print(f"Eventos históricos (event_ts < snapshot)        : {events_hist.count():,}")
print(f"Eventos futuros [snapshot, snapshot + {PREDICTION_HORIZON_DAYS}d) : {events_future.count():,}")

# COMMAND ----------
# DBTITLE 1,Filtro de elegibilidade: usuários inativos no corte

# Elegível = nenhum evento em [inactivity_cutoff, snapshot).
recent_active_users = (
    events.where(
        (F.col("event_ts") >= inactivity_ts) & (F.col("event_ts") < snapshot_ts)
    )
    .select("user_id")
    .distinct()
)

# Usuários com pelo menos 1 evento histórico em qualquer momento (precisamos
# de algum sinal pra calcular features).
users_with_history = events_hist.select("user_id").distinct()

eligible_users = users_with_history.join(
    recent_active_users, on="user_id", how="left_anti"
)

n_total_users = users.count()
n_with_history = users_with_history.count()
n_eligible = eligible_users.count()
print(f"Total de usuários                           : {n_total_users:,}")
print(f"Com algum evento histórico                  : {n_with_history:,}")
print(f"Elegíveis (inativos no corte) -> features   : {n_eligible:,}")

# COMMAND ----------
# DBTITLE 1,Calcular o label

label_df = (
    eligible_users.join(
        events_future.select("user_id").distinct().withColumn(
            "_has_future_event", F.lit(1)
        ),
        on="user_id",
        how="left",
    )
    .withColumn(
        LABEL_COLUMN,
        F.coalesce(F.col("_has_future_event"), F.lit(0)).cast("int"),
    )
    .drop("_has_future_event")
)

display(
    label_df.groupBy(LABEL_COLUMN).count().orderBy(LABEL_COLUMN)
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parte 3 — Engenharia de features
# MAGIC
# MAGIC Calculamos as features **apenas com eventos históricos**
# MAGIC (`event_ts < snapshot`). Para volume de centenas de milhares de eventos,
# MAGIC usamos PySpark — não Pandas.

# COMMAND ----------
# DBTITLE 1,Features de atividade (recência, frequência, conclusão)

events_hist_books = events_hist.join(books, on="book_id", how="left")

activity_features = (
    events_hist_books.groupBy("user_id")
    .agg(
        F.count("*").alias("total_events"),
        F.sum("session_minutes").alias("total_session_minutes"),
        F.avg("session_minutes").alias("avg_session_minutes"),
        F.countDistinct("book_id").alias("distinct_books"),
        F.avg("progress_pct").alias("avg_progress_pct"),
        F.avg(F.col("finished").cast("int")).alias("pct_finished"),
        F.avg(
            (F.col("format") == F.lit("audiobook")).cast("int")
        ).alias("pct_audiobook"),
        F.avg(
            (F.col("format") == F.lit("ebook")).cast("int")
        ).alias("pct_ebook"),
        F.max("event_ts").alias("_max_event_ts"),
    )
    .withColumn(
        "days_since_last_event",
        F.datediff(snapshot_ts.cast("date"), F.col("_max_event_ts").cast("date")),
    )
    .drop("_max_event_ts")
)

# COMMAND ----------
# DBTITLE 1,Top gênero por usuário

# Conta eventos por (user_id, genre) e mantém apenas o gênero líder de cada usuário.
genre_counts = (
    events_hist_books.where(F.col("genre").isNotNull())
    .groupBy("user_id", "genre")
    .agg(F.count("*").alias("n"))
)

genre_window = Window.partitionBy("user_id").orderBy(F.col("n").desc(), F.col("genre"))
top_genre = (
    genre_counts.withColumn("_rk", F.row_number().over(genre_window))
    .where(F.col("_rk") == 1)
    .select("user_id", F.col("genre").alias("top_genre"))
)

# COMMAND ----------
# DBTITLE 1,Janelas de eventos (90 e 180 dias antes do snapshot)

window_90_start = F.lit((snapshot - timedelta(days=90)).isoformat()).cast("timestamp")
window_180_start = F.lit((snapshot - timedelta(days=180)).isoformat()).cast("timestamp")

events_last_90d = (
    events_hist.where(F.col("event_ts") >= window_90_start)
    .groupBy("user_id")
    .agg(F.count("*").alias("events_last_90d"))
)
events_last_180d = (
    events_hist.where(F.col("event_ts") >= window_180_start)
    .groupBy("user_id")
    .agg(F.count("*").alias("events_last_180d"))
)

# COMMAND ----------
# DBTITLE 1,Features demográficas e de tenure (vindas de users)

user_demo = users.select(
    "user_id",
    F.datediff(snapshot_ts.cast("date"), F.col("signup_date")).alias("tenure_days"),
    "plan_type",
    "region",
    "device_type",
    "age_band",
)

# COMMAND ----------
# DBTITLE 1,Features de subscription

# had_subscription = true se o usuário aparece em subscriptions.
# subscription_canceled = true se já existiu alguma linha com end_date IS NOT NULL.
subs_features = (
    subs.groupBy("user_id")
    .agg(
        F.lit(True).alias("had_subscription"),
        F.max(F.col("end_date").isNotNull()).alias("subscription_canceled"),
    )
)

# COMMAND ----------
# DBTITLE 1,Montar a feature table final

df_features = (
    label_df  # já restrito aos elegíveis e com will_reactivate
    .join(activity_features, on="user_id", how="left")
    .join(top_genre, on="user_id", how="left")
    .join(events_last_90d, on="user_id", how="left")
    .join(events_last_180d, on="user_id", how="left")
    .join(user_demo, on="user_id", how="left")
    .join(subs_features, on="user_id", how="left")
)

# Tratamento de nulos para colunas numéricas de janela (usuários sem eventos
# na janela).
df_features = df_features.fillna(
    {
        "events_last_90d": 0,
        "events_last_180d": 0,
        "had_subscription": False,
        "subscription_canceled": False,
    }
)

# Reordena: chave, label, demais features
ordered_cols = [
    "user_id",
    LABEL_COLUMN,
    "tenure_days",
    "plan_type",
    "region",
    "device_type",
    "age_band",
    "total_events",
    "total_session_minutes",
    "avg_session_minutes",
    "distinct_books",
    "avg_progress_pct",
    "pct_finished",
    "pct_audiobook",
    "pct_ebook",
    "top_genre",
    "days_since_last_event",
    "events_last_90d",
    "events_last_180d",
    "had_subscription",
    "subscription_canceled",
]
df_features = df_features.select(*ordered_cols)

print(f"Linhas em customer_features: {df_features.count():,}")
print(f"Colunas: {len(df_features.columns)}")

# COMMAND ----------
# DBTITLE 1,Persistir customer_features em Delta

(
    df_features.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(config.table(TABLE_CUSTOMER_FEATURES))
)
print(f"OK: {config.table(TABLE_CUSTOMER_FEATURES)} salva em Delta.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Parte 4 — Sumário da feature table

# COMMAND ----------
# DBTITLE 1,Amostra das features

display(spark.sql(f"SELECT * FROM {config.table(TABLE_CUSTOMER_FEATURES)} LIMIT 20"))

# COMMAND ----------
# DBTITLE 1,Balanceamento das classes

display(
    spark.sql(
        f"""
        SELECT {LABEL_COLUMN},
               COUNT(*) AS n,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
        FROM {config.table(TABLE_CUSTOMER_FEATURES)}
        GROUP BY {LABEL_COLUMN}
        ORDER BY {LABEL_COLUMN}
        """
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC **Atenção — desbalanceamento esperado**: a classe positiva
# MAGIC (`will_reactivate = 1`) é **minoritária** por construção do problema.
# MAGIC Isso é normal em churn/reativação. O AutoML cuida disso automaticamente:
# MAGIC
# MAGIC - **Estratificação** no split treino/validação/teste.
# MAGIC - **Métricas adequadas** (F1, ROC AUC, PR AUC) em vez de acurácia.
# MAGIC - Treina vários algoritmos e seleciona o melhor pela métrica primária.
# MAGIC
# MAGIC Você pode ajustar a métrica primária no AutoML caso queira priorizar
# MAGIC precisão ou recall — veremos isso nos próximos notebooks.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Feature table pronta
# MAGIC
# MAGIC `customer_features` está disponível em `config.schema_full` com a coluna
# MAGIC de label `will_reactivate` e ~20 features de atividade, demografia e
# MAGIC assinatura.
# MAGIC
# MAGIC **Próximo passo**: rodar **um** dos dois notebooks de AutoML —
# MAGIC `03_automl_training/03a_automl_ui` (UI guiada) ou
# MAGIC `03_automl_training/03b_automl_sdk` (programático). Os dois produzem
# MAGIC um modelo equivalente; escolha de acordo com sua preferência.
