# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Model Serving + Inference Tables
# MAGIC
# MAGIC Este notebook cria (ou atualiza) um **endpoint Mosaic AI Model
# MAGIC Serving** servindo a versão atualmente apontada pelo alias
# MAGIC `Champion` do modelo registrado no notebook 04, com **Inference
# MAGIC Tables** ativadas para captura automática de requests/responses.
# MAGIC
# MAGIC ## Conceitos
# MAGIC
# MAGIC - **Model Serving** é uma plataforma serverless gerenciada que
# MAGIC   expõe modelos MLflow como endpoints REST com auto-scaling. O
# MAGIC   workspace cuida de provisionar compute, fazer o roll-out das
# MAGIC   versões e o roteamento de tráfego.
# MAGIC - **Workload size**: define a faixa de provisioned concurrency.
# MAGIC   `"Small"` corresponde a aproximadamente 4 requests concorrentes em
# MAGIC   regime — suficiente para esta demo.
# MAGIC - **Scale-to-zero**: se não houver tráfego, o endpoint desaloca o
# MAGIC   compute (custo zero), em troca de uma **cold start** de cerca de
# MAGIC   30 segundos na primeira chamada após a idle.
# MAGIC - **Inference Tables (via AI Gateway)**: o serving grava
# MAGIC   automaticamente cada request e response em uma tabela Delta no
# MAGIC   schema configurado, com nome `<schema>.<prefix>_payload`. Esse
# MAGIC   log é a base para o monitoring (notebook 07). O caminho legado
# MAGIC   `AutoCaptureConfigInput` foi descontinuado — usamos o `AiGatewayConfig`.
# MAGIC
# MAGIC ## Aviso de tempo
# MAGIC
# MAGIC `serving_endpoints.create_and_wait` pode levar **10 a 15 minutos** na
# MAGIC primeira criação (provisionar container, carregar modelo, health
# MAGIC checks). Atualizações subsequentes são mais rápidas.
# MAGIC
# MAGIC ## Notebook anterior
# MAGIC
# MAGIC `05_inference_notebook/05_inference_notebook`.
# MAGIC
# MAGIC ## Próximo notebook
# MAGIC
# MAGIC `07_monitoring/07_monitoring`.

# COMMAND ----------

# MAGIC %run ../config/demo_config

# COMMAND ----------
# DBTITLE 1,Carregar configuração

get_widgets(dbutils, spark)
config = resolve_config(dbutils, spark)

print(f"Endpoint name : {config.endpoint_name}")
print(f"Modelo (UC)   : {config.model_full_name}")
print(f"Inference Tbl : {config.table(INFERENCE_TABLE_PREFIX + '_payload')}")

# COMMAND ----------
# DBTITLE 1,Imports do Databricks SDK e MLflow

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
)
import mlflow

w = WorkspaceClient()

# COMMAND ----------
# DBTITLE 1,Resolver a versão atual do alias Champion

mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

champion = client.get_model_version_by_alias(
    name=config.model_full_name, alias=ALIAS_CHAMPION
)
print(f"Servindo {config.model_full_name} versão {champion.version} "
      f"(run_id={champion.run_id})")

# COMMAND ----------
# DBTITLE 1,Montar a configuração do endpoint

# `ServedEntityInput` declara qual modelo (entity_name + entity_version)
# rodar e como (workload_size + scale_to_zero_enabled). Demos do tipo
# CPU Small com scale-to-zero são o padrão custo-eficiente.
served_entity = ServedEntityInput(
    name="champion",
    entity_name=config.model_full_name,
    entity_version=str(champion.version),
    workload_size="Small",
    scale_to_zero_enabled=True,
)

# `AiGatewayConfig` ativa as Inference Tables via AI Gateway. (O caminho
# antigo via `AutoCaptureConfigInput` foi descontinuado: endpoints novos
# precisam usar AI Gateway.) O Databricks cria a tabela
# `<catalog>.<schema>.<table_name_prefix>_payload` automaticamente após a
# primeira chamada ao endpoint (latência de ~5-10 min até o primeiro
# flush — ver notebook 07).
ai_gateway = AiGatewayConfig(
    inference_table_config=AiGatewayInferenceTableConfig(
        catalog_name=config.catalog,
        schema_name=config.schema,
        table_name_prefix=INFERENCE_TABLE_PREFIX,
        enabled=True,
    )
)

# `EndpointCoreConfigInput` agrega só served_entities. A AI Gateway é
# passada como kwarg separado em create_and_wait/put_ai_gateway.
endpoint_config = EndpointCoreConfigInput(
    served_entities=[served_entity],
)

itc = ai_gateway.inference_table_config
print("Config preparada:")
print(f"  served_entities = [{served_entity.name} -> "
      f"{served_entity.entity_name} v{served_entity.entity_version}, "
      f"size={served_entity.workload_size}, "
      f"scale_to_zero={served_entity.scale_to_zero_enabled}]")
print(f"  ai_gateway.inference_table = catalog={itc.catalog_name}, "
      f"schema={itc.schema_name}, prefix={itc.table_name_prefix}, "
      f"enabled={itc.enabled}")

# COMMAND ----------
# DBTITLE 1,Criar ou atualizar o endpoint (idempotente)

# Idempotência: você pode rodar este notebook várias vezes (ex: após
# treinar uma versão nova). Se o endpoint já existe, fazemos `update`
# em vez de tentar criar de novo (que falharia com "already exists").
try:
    w.serving_endpoints.get(name=config.endpoint_name)
    endpoint_existe = True
    print(f"Endpoint '{config.endpoint_name}' já existe — vamos atualizar.")
except Exception as e:
    endpoint_existe = False
    print(f"Endpoint '{config.endpoint_name}' não encontrado — vamos criar.")
    print(f"  (Detalhe: {type(e).__name__}: {e})")

if endpoint_existe:
    # No update, served_entities e ai_gateway são aplicados em chamadas
    # separadas: `update_config_and_wait` para os modelos servidos e
    # `put_ai_gateway` para a configuração de Inference Tables.
    print("Aguardando update do endpoint... (pode levar alguns minutos)")
    w.serving_endpoints.update_config_and_wait(
        name=config.endpoint_name,
        served_entities=endpoint_config.served_entities,
    )
    w.serving_endpoints.put_ai_gateway(
        name=config.endpoint_name,
        inference_table_config=ai_gateway.inference_table_config,
    )
    print(f"Endpoint '{config.endpoint_name}' atualizado.")
else:
    print("Aguardando criação do endpoint... pode levar 10-15 minutos.")
    w.serving_endpoints.create_and_wait(
        name=config.endpoint_name,
        config=endpoint_config,
        ai_gateway=ai_gateway,
    )
    print(f"Endpoint '{config.endpoint_name}' criado e pronto.")

# COMMAND ----------
# DBTITLE 1,Status final do endpoint

ep = w.serving_endpoints.get(name=config.endpoint_name)
state = getattr(ep, "state", None)
print(f"Endpoint    : {ep.name}")
print(f"State.ready : {getattr(state, 'ready', None)}")
print(f"State.config_update : {getattr(state, 'config_update', None)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Chamada de teste ao endpoint
# MAGIC
# MAGIC Vamos enviar 3 linhas (sem `user_id` e sem o label) ao endpoint via
# MAGIC `WorkspaceClient.serving_endpoints.query()` — o método canônico do
# MAGIC Databricks SDK para chamar Model Serving. Ele cuida da autenticação
# MAGIC (host e token vêm do contexto do notebook) e da serialização do
# MAGIC payload `dataframe_split`, formato aceito por modelos pyfunc /
# MAGIC sklearn / LightGBM (o que o AutoML produz por padrão).

# COMMAND ----------
# DBTITLE 1,Chamada via Databricks SDK

sample = (
    spark.table(config.table(TABLE_CUSTOMER_FEATURES))
    .drop("user_id", LABEL_COLUMN)
    .limit(3)
    .toPandas()
)
print(f"Amostra: {sample.shape}, cols={list(sample.columns)}")

split = sample.to_dict(orient="split")

# `dataframe_split` é o formato canônico para modelos pyfunc/tabulares.
# O SDK monta a request, autentica e devolve a resposta já parseada.
response = w.serving_endpoints.query(
    name=config.endpoint_name,
    dataframe_split={
        "columns": split["columns"],
        "data": split["data"],
    },
)

print("Predictions:", response.predictions)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Endpoint servindo o modelo Champion
# MAGIC
# MAGIC O que está rodando agora:
# MAGIC
# MAGIC - Endpoint `config.endpoint_name`, CPU Small, scale-to-zero habilitado.
# MAGIC - Versão servida: a apontada pelo alias `Champion` no momento do
# MAGIC   `create`/`update`. Para apontar para uma versão nova, mova o alias
# MAGIC   no notebook 04 e re-execute este notebook.
# MAGIC - Inference Tables ativadas: a tabela
# MAGIC   `<catalog>.<schema>.inference_log_payload` será criada/atualizada
# MAGIC   automaticamente.
# MAGIC
# MAGIC ## Atenção — latência das Inference Tables
# MAGIC
# MAGIC O serving captura cada request, mas o flush para a tabela Delta tem
# MAGIC latência de **~5 a 10 minutos**. Se você for direto ao notebook
# MAGIC `07_monitoring`, pode encontrar a tabela vazia ou inexistente.
# MAGIC Espere alguns minutos antes de seguir.
# MAGIC
# MAGIC ## Não esqueça do teardown
# MAGIC
# MAGIC Mesmo com scale-to-zero há um custo mínimo associado a manter o
# MAGIC endpoint provisionado. Rode o notebook `99_teardown` ao terminar.
# MAGIC
# MAGIC **Próximo passo**: rodar `07_monitoring/07_monitoring`.
