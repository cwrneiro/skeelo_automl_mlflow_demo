# Databricks notebook source
# MAGIC %md
# MAGIC # Configuração central da demo
# MAGIC
# MAGIC Este notebook é carregado via `%run ../config/demo_config` em todos os
# MAGIC outros notebooks da demo. Ele define **constantes de contrato**, a
# MAGIC dataclass `DemoConfig` e helpers para criar/ler widgets.
# MAGIC
# MAGIC **Não modifique este arquivo no meio da demo** — alterações aqui são
# MAGIC contrato global e exigem revisão dos notebooks que dependem dele.

# COMMAND ----------

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
import os
import re

# COMMAND ----------
# DBTITLE 1,Constantes de contrato

# Tabelas brutas (geradas em 01_data_generation, no seu schema)
TABLE_USERS = "users"
TABLE_BOOKS = "books"
TABLE_READING_EVENTS = "reading_events"
TABLE_SUBSCRIPTIONS = "subscriptions"

# Feature table derivada (construída em 02_eda) — entrada do AutoML
TABLE_CUSTOMER_FEATURES = "customer_features"

# Coluna de label binária no dataset de treino
LABEL_COLUMN = "will_reactivate"

# Modelo registrado em Unity Catalog: <catalog>.<schema>.<MODEL_NAME>
MODEL_NAME = "reactivation_model"

# Aliases MLflow no UC Model Registry
ALIAS_CHAMPION = "Champion"
ALIAS_CHALLENGER = "Challenger"

# Reprodutibilidade dos dados mock
RANDOM_SEED = 42

# Volume da geração mock
N_USERS = 10_000
N_EVENTS_TARGET = 500_000

# Janela do label
INACTIVITY_WINDOW_DAYS = 30
PREDICTION_HORIZON_DAYS = 30

# Inference Tables — prefix usado no endpoint de Model Serving
INFERENCE_TABLE_PREFIX = "inference_log"

# COMMAND ----------
# DBTITLE 1,Estrutura de configuração

@dataclass(frozen=True)
class DemoConfig:
    catalog: str
    schema: str
    endpoint_name: str
    snapshot_date: date
    user_short: str

    @property
    def schema_full(self) -> str:
        return f"{self.catalog}.{self.schema}"

    def table(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    @property
    def model_full_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{MODEL_NAME}"

# COMMAND ----------
# DBTITLE 1,Helpers internos

def _short_user(email: str) -> str:
    """Deriva um identificador curto e seguro para nomes de schema/endpoint."""
    local = email.split("@")[0]
    return re.sub(r"[^a-z0-9]+", "_", local.lower()).strip("_")[:24]


def _current_user(spark) -> str:
    return spark.sql("SELECT current_user() AS u").first()["u"]


def _config_file_path(spark) -> str:
    """Caminho do arquivo de configuração persistida (no seu Workspace).

    O arquivo é gravado por `00_setup` e lido pelos demais notebooks como
    *default* dos widgets, evitando que você precise repreencher a
    configuração em cada notebook.
    """
    email = _current_user(spark)
    return f"/Workspace/Users/{email}/.automl_demo_config.json"


def _load_saved_config(spark) -> dict:
    try:
        with open(_config_file_path(spark), "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, OSError, ValueError):
        return {}


def save_config(config: "DemoConfig", spark) -> str:
    """Persiste a configuração no seu Workspace (chamado por `00_setup`).

    Os demais notebooks usam esse arquivo como *default* ao criar widgets.
    Retorna o caminho do arquivo gravado.
    """
    path = _config_file_path(spark)
    payload = {
        "catalog": config.catalog,
        "schema": config.schema,
        "endpoint_name": config.endpoint_name,
        "snapshot_date": config.snapshot_date.isoformat(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path

# COMMAND ----------
# DBTITLE 1,Widgets e resolução

def get_widgets(dbutils, spark) -> None:
    """Cria os widgets padrão da demo. Chamar na primeira célula de código.

    Usa a configuração persistida pelo `00_setup` (se existir) como *default*.
    Assim, após o setup inicial, basta abrir os demais notebooks e rodar.
    """
    user_short = _short_user(_current_user(spark))
    saved = _load_saved_config(spark)
    dbutils.widgets.text(
        "catalog",
        saved.get("catalog", ""),
        "Catálogo Unity Catalog",
    )
    dbutils.widgets.text(
        "schema",
        saved.get("schema", f"automl_demo_{user_short}"),
        "Seu schema",
    )
    dbutils.widgets.text(
        "endpoint_name",
        saved.get("endpoint_name", f"automl-reactivation-{user_short}"),
        "Nome do endpoint Model Serving",
    )
    dbutils.widgets.text(
        "snapshot_date",
        saved.get("snapshot_date", ""),
        "Data de corte (YYYY-MM-DD; vazio = auto)",
    )


def resolve_config(dbutils, spark) -> DemoConfig:
    """Lê os widgets e devolve um `DemoConfig` validado."""
    catalog = dbutils.widgets.get("catalog").strip()
    if not catalog:
        raise ValueError(
            "Widget 'catalog' obrigatório. Informe o catálogo Unity Catalog "
            "onde o schema da demo deve ser criado e rode o notebook "
            "`00_setup` primeiro — ele persiste a configuração para os "
            "demais notebooks."
        )
    schema = dbutils.widgets.get("schema").strip()
    endpoint_name = dbutils.widgets.get("endpoint_name").strip()
    snapshot_raw = dbutils.widgets.get("snapshot_date").strip()

    if snapshot_raw:
        snapshot = date.fromisoformat(snapshot_raw)
    else:
        snapshot = date.today() - timedelta(days=INACTIVITY_WINDOW_DAYS)

    return DemoConfig(
        catalog=catalog,
        schema=schema,
        endpoint_name=endpoint_name,
        snapshot_date=snapshot,
        user_short=_short_user(_current_user(spark)),
    )
