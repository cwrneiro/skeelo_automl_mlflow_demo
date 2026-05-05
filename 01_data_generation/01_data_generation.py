# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Geração de dados mock
# MAGIC
# MAGIC Este notebook cria as **quatro tabelas Delta brutas** da demo, todas no
# MAGIC schema do participante (`config.schema_full`):
# MAGIC
# MAGIC - `users` — cadastro de clientes (~10.000 linhas).
# MAGIC - `books` — catálogo de livros/audiobooks (~2.000 linhas).
# MAGIC - `reading_events` — eventos de leitura/escuta (~500.000 linhas).
# MAGIC - `subscriptions` — assinaturas dos clientes premium.
# MAGIC
# MAGIC **Determinismo**: a geração usa `numpy.random.default_rng(RANDOM_SEED)`
# MAGIC com seed fixa (`RANDOM_SEED = 42`). Todos os participantes da demo geram
# MAGIC exatamente o mesmo dataset, o que garante resultados comparáveis em
# MAGIC AutoML, modelo, métricas e inferência.
# MAGIC
# MAGIC **Janela temporal**: eventos distribuídos pelos últimos ~24 meses até
# MAGIC **hoje**, com sazonalidade (boost em dezembro, janeiro e julho).
# MAGIC
# MAGIC **Label `will_reactivate`**: o label *não* é gerado aqui — ele é
# MAGIC calculado em `02_eda` a partir dos eventos. A geração é feita de modo
# MAGIC que naturalmente exista uma proporção realista de usuários inativos no
# MAGIC corte (~30%) e, dentre esses, uma fração que reativa (~25%).
# MAGIC
# MAGIC **Tempo alvo**: < 2 minutos em cluster single-node típico DBR 17.3 LTS ML.
# MAGIC
# MAGIC **Notebook anterior**: `00_setup/00_setup`.
# MAGIC
# MAGIC **Próximo notebook**: `02_eda/02_eda`.

# COMMAND ----------

# MAGIC %run ../config/demo_config

# COMMAND ----------
# DBTITLE 1,Carregar configuração

get_widgets(dbutils, spark)
config = resolve_config(dbutils, spark)
print(f"Schema alvo: {config.schema_full}")
print(f"Seed: {RANDOM_SEED}  N_USERS: {N_USERS}  N_EVENTS_TARGET: {N_EVENTS_TARGET}")

# COMMAND ----------
# DBTITLE 1,Selecionar catálogo e schema

spark.sql(f"USE CATALOG {config.catalog}")
spark.sql(f"USE SCHEMA {config.schema}")

# COMMAND ----------
# DBTITLE 1,Imports e RNG

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    BooleanType,
    DateType,
    TimestampType,
)

rng = np.random.default_rng(RANDOM_SEED)

# Data base = hoje. Toda a janela de 24 meses termina aqui.
TODAY = date.today()
WINDOW_DAYS = 24 * 30  # ~720 dias

print(f"Data base (hoje): {TODAY}")
print(f"Janela: {WINDOW_DAYS} dias até hoje")

# COMMAND ----------
# DBTITLE 1,Gerar users

n_users = N_USERS
user_ids = [f"u_{i:06d}" for i in range(n_users)]

# signup_date: distribuído nos últimos ~24 meses, com mais signups recentes.
# Usamos uma distribuição triangular favorecendo o presente.
signup_offsets = rng.triangular(
    left=0, mode=WINDOW_DAYS, right=WINDOW_DAYS, size=n_users
)
# offset 0 = mais antigo (hoje - 720), offset 720 = hoje
signup_dates = [
    TODAY - timedelta(days=int(WINDOW_DAYS - o)) for o in signup_offsets
]

# plan_type: 70% free / 30% premium
plan_type = rng.choice(["free", "premium"], size=n_users, p=[0.70, 0.30])

# age_band com pesos realistas
age_bands = ["<25", "25-34", "35-44", "45-54", "55+"]
age_weights = [0.18, 0.32, 0.25, 0.15, 0.10]
age_band = rng.choice(age_bands, size=n_users, p=age_weights)

# region: 5 regiões fictícias do Brasil
regions = ["Norte", "Sul", "Sudeste", "Nordeste", "Centro-Oeste"]
region_weights = [0.08, 0.15, 0.45, 0.22, 0.10]
region = rng.choice(regions, size=n_users, p=region_weights)

# device_type: 60/25/15
devices = ["mobile", "tablet", "desktop"]
device_weights = [0.60, 0.25, 0.15]
device_type = rng.choice(devices, size=n_users, p=device_weights)

users_pdf = pd.DataFrame(
    {
        "user_id": user_ids,
        "signup_date": signup_dates,
        "plan_type": plan_type,
        "age_band": age_band,
        "region": region,
        "device_type": device_type,
    }
)
print(f"users gerados: {len(users_pdf):,}")
users_pdf.head()

# COMMAND ----------
# DBTITLE 1,Gerar books

n_books = 2_000
book_ids = [f"b_{i:05d}" for i in range(n_books)]

# Título fictício simples: "Livro <adj> <noun> <num>"
adjectives = [
    "Silencioso", "Eterno", "Brilhante", "Oculto", "Antigo",
    "Distante", "Profundo", "Sereno", "Luminoso", "Selvagem",
]
nouns = [
    "Caminho", "Sonho", "Horizonte", "Mistério", "Oceano",
    "Jardim", "Reino", "Eco", "Refúgio", "Labirinto",
]
titles = [
    f"O {rng.choice(adjectives)} {rng.choice(nouns)} {rng.integers(1, 9999)}"
    for _ in range(n_books)
]

genres = [
    "Ficção", "Não-ficção", "Romance", "Suspense",
    "Fantasia", "Biografia", "Autoajuda", "Negócios",
]
genre_weights = [0.18, 0.14, 0.16, 0.13, 0.12, 0.10, 0.10, 0.07]
genre = rng.choice(genres, size=n_books, p=genre_weights)

# format: 60% ebook, 40% audiobook
fmt = rng.choice(["ebook", "audiobook"], size=n_books, p=[0.60, 0.40])

# length_minutes: normal truncado em [60, 1500]
lengths = rng.normal(loc=520, scale=260, size=n_books)
lengths = np.clip(lengths, 60, 1500).astype(int)

# language: pt dominante
languages = rng.choice(["pt", "en", "es"], size=n_books, p=[0.78, 0.16, 0.06])

books_pdf = pd.DataFrame(
    {
        "book_id": book_ids,
        "title": titles,
        "genre": genre,
        "format": fmt,
        "length_minutes": lengths,
        "language": languages,
    }
)
print(f"books gerados: {len(books_pdf):,}")
books_pdf.head()

# COMMAND ----------
# DBTITLE 1,Gerar reading_events

# Estratégia:
# - Distribuição Pareto de atividade por usuário (poucos muito ativos, maioria leve).
# - Cada evento tem timestamp dentro da janela [signup_date, hoje], com boost
#   de sazonalidade nos meses 12, 1 e 7.
# - ~30% dos usuários terão seu último evento "antigo" (>30 dias antes de hoje)
#   para serem candidatos a inativos. Dentro desses, ~25% terão pelo menos um
#   evento "recente" (últimos 30 dias) para representar reativação.
# - Não forçamos o label aqui; o label é derivado em 02_eda.

n_events_target = N_EVENTS_TARGET

# Pareto shape: alpha menor = cauda mais pesada (alguns usuários muito ativos).
# Normalizamos os pesos para somar n_events_target.
pareto_raw = rng.pareto(a=1.4, size=n_users) + 1.0  # >= 1
weights = pareto_raw / pareto_raw.sum()
events_per_user = np.round(weights * n_events_target).astype(int)
# Garante mínimo 1 evento para a maioria (alguns terão 0 — usuários "frios")
# Forçamos ~5% dos usuários a terem 0 eventos para representar churners totais.
zero_mask = rng.random(n_users) < 0.05
events_per_user[zero_mask] = 0
# Para os demais com 0, garante pelo menos 1
events_per_user[(events_per_user == 0) & (~zero_mask)] = 1

n_events_actual = int(events_per_user.sum())
print(f"Eventos planejados: {n_events_actual:,}")

# Decide quem é candidato a "inativo no corte" (~30%) e dentro disso quem
# "reativa" (~25%). Esse split orienta como vamos posicionar os timestamps,
# mas o label final será calculado a partir dos eventos em 02_eda.
inactive_candidate = rng.random(n_users) < 0.30
reactivator = inactive_candidate & (rng.random(n_users) < 0.25)

# Pré-aloca arrays
total = n_events_actual
ev_user_idx = np.empty(total, dtype=np.int64)
ev_offset_days = np.empty(total, dtype=np.float64)  # offset desde TODAY (negativo = passado)

# Preenche user index para cada evento
cursor = 0
for u_idx in range(n_users):
    k = events_per_user[u_idx]
    if k == 0:
        continue
    ev_user_idx[cursor : cursor + k] = u_idx
    cursor += k

# Para cada evento, gera offset (em dias antes de hoje) considerando:
# - signup do usuário (não pode ser anterior ao signup)
# - se inativo: a maioria dos eventos do usuário fica em [60, 720] dias atrás
# - se reativador: mantém a cauda antiga + injeta 1-3 eventos nos últimos 30 dias
# - se ativo (não inativo): distribuição uniforme com leve boost recente

signup_offsets_today = np.array(
    [(TODAY - sd).days for sd in signup_dates], dtype=np.int64
)  # dias entre signup e hoje (>= 0)

cursor = 0
for u_idx in range(n_users):
    k = events_per_user[u_idx]
    if k == 0:
        continue
    max_age = min(int(signup_offsets_today[u_idx]), WINDOW_DAYS)
    if max_age < 1:
        max_age = 1

    if inactive_candidate[u_idx]:
        # Maioria dos eventos em [60, max_age]
        lo = min(60, max_age - 1)
        offsets = rng.uniform(lo, max_age, size=k)
        if reactivator[u_idx]:
            # Injeta 1-3 eventos nos últimos 30 dias (offset 0..29)
            n_recent = int(rng.integers(1, 4))
            n_recent = min(n_recent, k)
            recent = rng.uniform(0, 29, size=n_recent)
            offsets[:n_recent] = recent
    else:
        # Usuário ativo: distribuição uniforme com leve concentração nos
        # últimos 90 dias (50% recente / 50% antigo).
        n_recent = int(k * 0.5)
        n_old = k - n_recent
        recent = rng.uniform(0, min(90, max_age), size=n_recent)
        old = rng.uniform(0, max_age, size=n_old) if n_old > 0 else np.array([])
        offsets = np.concatenate([recent, old]) if n_old > 0 else recent

    ev_offset_days[cursor : cursor + k] = offsets
    cursor += k

# Aplica sazonalidade: rejeita-aceita simples sobre o mês do offset.
# Boost: dezembro, janeiro, julho.
seasonal_boost = {12: 1.5, 1: 1.4, 7: 1.3}

def _seasonal_keep(offsets):
    # Para cada evento, calcula mês e aceita com prob proporcional ao boost.
    # Probabilidade base = 1.0; meses boostados têm prob > 1, então truncamos
    # via reescala — eventos NÃO boostados são potencialmente substituídos.
    # Implementação simples: reamostra ~15% dos eventos para meses boostados.
    n = len(offsets)
    months = np.array(
        [(TODAY - timedelta(days=int(o))).month for o in offsets]
    )
    # Probabilidade de manter; depois reamostraremos o resto para meses boostados
    keep_prob = np.array([0.85] * n)
    for m, _ in seasonal_boost.items():
        keep_prob[months == m] = 1.0
    keep = rng.random(n) < keep_prob
    return keep

# Em vez de rejeitar, aplicamos um "shift" que move parte dos eventos não
# boostados para datas próximas em meses boostados. Mais simples: para ~10%
# dos eventos, recoloca em offset que cai num mês boostado.
shift_mask = rng.random(total) < 0.10
if shift_mask.any():
    # Para cada evento marcado, escolhe um mês boostado e um offset cujo mês bata.
    boosted_months = list(seasonal_boost.keys())
    # Pré-computa, para cada offset em 0..WINDOW_DAYS, o mês correspondente.
    all_offsets = np.arange(0, WINDOW_DAYS + 1)
    all_months = np.array(
        [(TODAY - timedelta(days=int(o))).month for o in all_offsets]
    )
    boosted_pool = all_offsets[np.isin(all_months, boosted_months)]
    if len(boosted_pool) > 0:
        n_shift = int(shift_mask.sum())
        # Respeitar limites por usuário: para cada evento alvo, sorteia um
        # offset boostado e clampa pelo signup do usuário.
        sampled = rng.choice(boosted_pool, size=n_shift, replace=True)
        target_idx = np.where(shift_mask)[0]
        # Clampa por signup do usuário
        u_for_target = ev_user_idx[target_idx]
        max_for_target = signup_offsets_today[u_for_target]
        sampled = np.minimum(sampled, np.maximum(max_for_target, 0))
        ev_offset_days[target_idx] = sampled

# Converte offset em timestamp: TODAY - offset_dias + hora aleatória
hours = rng.integers(0, 24, size=total)
minutes = rng.integers(0, 60, size=total)
seconds = rng.integers(0, 60, size=total)

base_dt = datetime(TODAY.year, TODAY.month, TODAY.day)
event_ts = [
    base_dt
    - timedelta(days=int(ev_offset_days[i]))
    + timedelta(
        hours=int(hours[i]), minutes=int(minutes[i]), seconds=int(seconds[i])
    )
    for i in range(total)
]

# book_id: amostragem uniforme do catálogo
book_idx = rng.integers(0, n_books, size=total)
event_books = [book_ids[i] for i in book_idx]

# progress_pct: massa em 0.0-0.1 (abandono cedo) e em 1.0 (terminou)
mix = rng.random(total)
progress = np.empty(total, dtype=np.float64)
early_mask = mix < 0.45
finish_mask = (mix >= 0.45) & (mix < 0.70)
mid_mask = mix >= 0.70
progress[early_mask] = rng.uniform(0.0, 0.10, size=int(early_mask.sum()))
progress[finish_mask] = rng.uniform(0.95, 1.00, size=int(finish_mask.sum()))
progress[mid_mask] = rng.uniform(0.10, 0.95, size=int(mid_mask.sum()))

# session_minutes: 1-180, log-normal-ish
session = rng.gamma(shape=2.0, scale=15.0, size=total)
session = np.clip(session, 1, 180).astype(int)

# finished correlacionado com progress >= 0.95
finished = progress >= 0.95

# rating: null em ~80%; quando preenchido, 1-5 com peso para 4-5
rating = np.full(total, np.nan, dtype=np.float64)
rated_mask = rng.random(total) < 0.20
rating_values = rng.choice(
    [1, 2, 3, 4, 5], size=int(rated_mask.sum()), p=[0.05, 0.07, 0.18, 0.35, 0.35]
)
rating[rated_mask] = rating_values

# Mapeia user index -> user_id
event_users = [user_ids[i] for i in ev_user_idx]

events_pdf = pd.DataFrame(
    {
        "user_id": event_users,
        "book_id": event_books,
        "event_ts": event_ts,
        "progress_pct": progress,
        "session_minutes": session,
        "finished": finished,
        "rating": rating,  # pandas float com NaN; convertido para Int abaixo
    }
)

# Converte rating em Int nullable para preservar NULLs no Spark
events_pdf["rating"] = events_pdf["rating"].astype("Int64")

print(f"reading_events gerados: {len(events_pdf):,}")
print(f"  faixa de event_ts: {events_pdf['event_ts'].min()}  ->  {events_pdf['event_ts'].max()}")
print(f"  finished == True: {int(events_pdf['finished'].sum()):,}")
print(f"  rating não-nulos: {int(events_pdf['rating'].notna().sum()):,}")
events_pdf.head()

# COMMAND ----------
# DBTITLE 1,Gerar subscriptions

# Apenas usuários premium têm linha em subscriptions.
# - 1 ou mais assinaturas por usuário (a maioria 1).
# - start_date próximo do signup_date.
# - ~30% canceladas (end_date preenchido + cancel_reason).
# - As demais ativas (end_date = NULL, cancel_reason = NULL).

premium_mask = users_pdf["plan_type"].values == "premium"
premium_user_ids = users_pdf.loc[premium_mask, "user_id"].tolist()
premium_signup = users_pdf.loc[premium_mask, "signup_date"].tolist()
n_premium = len(premium_user_ids)

cancel_reasons = [
    "preço", "pouco uso", "trocou de plataforma",
    "conteúdo insuficiente", "problema técnico", "outro",
]

sub_rows = []
for u_id, sd in zip(premium_user_ids, premium_signup):
    # 80% têm 1 assinatura, 15% têm 2, 5% têm 3
    n_subs = int(rng.choice([1, 2, 3], p=[0.80, 0.15, 0.05]))
    last_end = sd
    for s_i in range(n_subs):
        # start_date: próximo do signup ou do fim da assinatura anterior
        offset_start = int(rng.integers(0, 30))
        start = last_end + timedelta(days=offset_start)
        if start > TODAY:
            start = TODAY - timedelta(days=int(rng.integers(1, 30)))
        # ~30% canceladas
        canceled = rng.random() < 0.30
        if canceled:
            duration = int(rng.integers(30, 540))
            end = start + timedelta(days=duration)
            if end > TODAY:
                end = TODAY
            reason = str(rng.choice(cancel_reasons))
            sub_rows.append((u_id, start, end, reason))
            last_end = end
        else:
            sub_rows.append((u_id, start, None, None))
            last_end = TODAY  # ativa, não encadeia outra
            break  # uma ativa basta

subs_pdf = pd.DataFrame(
    sub_rows, columns=["user_id", "start_date", "end_date", "cancel_reason"]
)
print(f"subscriptions geradas: {len(subs_pdf):,}  (usuários premium: {n_premium:,})")
subs_pdf.head()

# COMMAND ----------
# DBTITLE 1,Schemas Spark

users_schema = StructType(
    [
        StructField("user_id", StringType(), False),
        StructField("signup_date", DateType(), False),
        StructField("plan_type", StringType(), False),
        StructField("age_band", StringType(), False),
        StructField("region", StringType(), False),
        StructField("device_type", StringType(), False),
    ]
)

books_schema = StructType(
    [
        StructField("book_id", StringType(), False),
        StructField("title", StringType(), False),
        StructField("genre", StringType(), False),
        StructField("format", StringType(), False),
        StructField("length_minutes", IntegerType(), False),
        StructField("language", StringType(), False),
    ]
)

events_schema = StructType(
    [
        StructField("user_id", StringType(), False),
        StructField("book_id", StringType(), False),
        StructField("event_ts", TimestampType(), False),
        StructField("progress_pct", DoubleType(), False),
        StructField("session_minutes", IntegerType(), False),
        StructField("finished", BooleanType(), False),
        StructField("rating", IntegerType(), True),
    ]
)

subs_schema = StructType(
    [
        StructField("user_id", StringType(), False),
        StructField("start_date", DateType(), False),
        StructField("end_date", DateType(), True),
        StructField("cancel_reason", StringType(), True),
    ]
)

# COMMAND ----------
# DBTITLE 1,Escrever users em Delta

# Converte rating Int64 nullable para object com None (compatível com Spark IntegerType)
def _to_spark_safe(pdf: pd.DataFrame) -> pd.DataFrame:
    out = pdf.copy()
    for col in out.columns:
        if str(out[col].dtype) == "Int64":
            out[col] = out[col].astype(object).where(out[col].notna(), None)
    return out

users_sdf = spark.createDataFrame(_to_spark_safe(users_pdf), schema=users_schema)
users_sdf.write.mode("overwrite").saveAsTable(config.table(TABLE_USERS))
print(f"OK: {config.table(TABLE_USERS)} ({users_sdf.count():,} linhas)")

# COMMAND ----------
# DBTITLE 1,Escrever books em Delta

books_sdf = spark.createDataFrame(_to_spark_safe(books_pdf), schema=books_schema)
books_sdf.write.mode("overwrite").saveAsTable(config.table(TABLE_BOOKS))
print(f"OK: {config.table(TABLE_BOOKS)} ({books_sdf.count():,} linhas)")

# COMMAND ----------
# DBTITLE 1,Escrever reading_events em Delta

events_sdf = spark.createDataFrame(_to_spark_safe(events_pdf), schema=events_schema)
events_sdf.write.mode("overwrite").saveAsTable(config.table(TABLE_READING_EVENTS))
print(f"OK: {config.table(TABLE_READING_EVENTS)} ({events_sdf.count():,} linhas)")

# COMMAND ----------
# DBTITLE 1,Escrever subscriptions em Delta

subs_sdf = spark.createDataFrame(_to_spark_safe(subs_pdf), schema=subs_schema)
subs_sdf.write.mode("overwrite").saveAsTable(config.table(TABLE_SUBSCRIPTIONS))
print(f"OK: {config.table(TABLE_SUBSCRIPTIONS)} ({subs_sdf.count():,} linhas)")

# COMMAND ----------
# DBTITLE 1,Sumário e amostras

print("Contagens finais:")
for tname in [TABLE_USERS, TABLE_BOOKS, TABLE_READING_EVENTS, TABLE_SUBSCRIPTIONS]:
    n = spark.sql(f"SELECT COUNT(*) AS n FROM {config.table(tname)}").first()["n"]
    print(f"  {tname:20s}  {n:>10,d}")

max_ts = spark.sql(
    f"SELECT MAX(event_ts) AS m FROM {config.table(TABLE_READING_EVENTS)}"
).first()["m"]
min_ts = spark.sql(
    f"SELECT MIN(event_ts) AS m FROM {config.table(TABLE_READING_EVENTS)}"
).first()["m"]
print(f"\nFaixa de event_ts: {min_ts}  ->  {max_ts}")
print(
    "Sugestão: use snapshot_date próxima de "
    f"{(max_ts.date() - timedelta(days=INACTIVITY_WINDOW_DAYS)) if max_ts else 'N/A'} "
    "para que existam usuários inativos no corte."
)

# COMMAND ----------
# DBTITLE 1,Amostra de users

display(spark.sql(f"SELECT * FROM {config.table(TABLE_USERS)} LIMIT 10"))

# COMMAND ----------
# DBTITLE 1,Amostra de books

display(spark.sql(f"SELECT * FROM {config.table(TABLE_BOOKS)} LIMIT 10"))

# COMMAND ----------
# DBTITLE 1,Amostra de reading_events

display(
    spark.sql(
        f"SELECT * FROM {config.table(TABLE_READING_EVENTS)} ORDER BY event_ts DESC LIMIT 10"
    )
)

# COMMAND ----------
# DBTITLE 1,Amostra de subscriptions

display(spark.sql(f"SELECT * FROM {config.table(TABLE_SUBSCRIPTIONS)} LIMIT 10"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Dados gerados
# MAGIC
# MAGIC As quatro tabelas brutas estão disponíveis em `config.schema_full`:
# MAGIC `users`, `books`, `reading_events`, `subscriptions`.
# MAGIC
# MAGIC **Próximo passo**: rodar `02_eda/02_eda` para a análise exploratória,
# MAGIC construção do label `will_reactivate` e da tabela de features
# MAGIC `customer_features`.
