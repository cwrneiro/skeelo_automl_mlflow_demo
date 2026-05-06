# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Registro do modelo no Unity Catalog
# MAGIC
# MAGIC Este notebook **registra no Unity Catalog Model Registry** o melhor
# MAGIC modelo produzido pelo AutoML (notebook `03a_automl_ui` ou
# MAGIC `03b_automl_sdk`), atribui o alias **Champion** e demonstra o padrão
# MAGIC **Champion / Challenger**.
# MAGIC
# MAGIC ## O que é o UC Model Registry
# MAGIC
# MAGIC No Unity Catalog **modelos são objetos UC de primeira classe**, com o
# MAGIC mesmo modelo de governança das tabelas: catálogo / schema / nome,
# MAGIC permissões `USE`, `EXECUTE`, `MANAGE`, lineage etc. O nome canônico do
# MAGIC modelo desta demo é `config.model_full_name`
# MAGIC (`<catalog>.<schema>.reactivation_model`).
# MAGIC
# MAGIC Cada chamada de `mlflow.register_model(...)` cria uma **nova versão**
# MAGIC monotônica (1, 2, 3, ...). Em vez de `stages` (`Staging`, `Production`),
# MAGIC que existiam no antigo Workspace Model Registry, o UC usa **aliases**
# MAGIC livres e movíveis. Convenção da demo:
# MAGIC
# MAGIC - `Champion` → versão atualmente em produção (servida pelo endpoint).
# MAGIC - `Challenger` → próxima versão candidata, em avaliação A/B.
# MAGIC
# MAGIC ## Notebook anterior
# MAGIC
# MAGIC `03_automl_training/03a_automl_ui` ou `03b_automl_sdk` — produziram um
# MAGIC `run_id` MLflow do best trial.
# MAGIC
# MAGIC ## Próximo notebook
# MAGIC
# MAGIC `05_inference_notebook/05_inference_notebook`.

# COMMAND ----------

# MAGIC %run ../config/demo_config

# COMMAND ----------
# DBTITLE 1,Carregar configuração e widget extra

get_widgets(dbutils, spark)

# Widget extra: run_id do best trial. Se vazio, fazemos a busca automática
# do run mais recente do experimento mais recente do user_short.
dbutils.widgets.text("run_id", "", "MLflow run_id (vazio = auto)")

config = resolve_config(dbutils, spark)
run_id_widget = dbutils.widgets.get("run_id").strip()

print(f"Modelo alvo  : {config.model_full_name}")
print(f"user_short   : {config.user_short}")
print(f"run_id widget: '{run_id_widget}' (vazio => busca automática)")

# COMMAND ----------
# DBTITLE 1,Configurar MLflow para Unity Catalog

import mlflow

# Esta linha é o que muda do registro **workspace-level** para o
# **Unity Catalog Model Registry**. Sem ela, `mlflow.register_model` cairia
# no registry antigo (workspace), que não compartilha governança com UC.
mlflow.set_registry_uri("databricks-uc")

client = mlflow.MlflowClient()
print("Registry URI:", mlflow.get_registry_uri())

# COMMAND ----------
# DBTITLE 1,Resolver o run_id (manual ou automático)

def _auto_pick_run_id(client: "mlflow.MlflowClient", user_short: str) -> str:
    """Busca o run mais recente do experimento mais recente do user_short.

    Estratégia:
    1. Lista experimentos cujo nome contém o user_short.
    2. Pega o experimento com `last_update_time` maior.
    3. Dentro dele, ordena os runs pela métrica primária (ou por start_time
       se a métrica não estiver disponível) e devolve o melhor.
    """
    exps = client.search_experiments(
        filter_string=f"name LIKE '%{user_short}%'"
    )
    if not exps:
        raise RuntimeError(
            "Nenhum experimento MLflow encontrado contendo "
            f"'{user_short}' no nome.\n"
            "  - Rode antes 03a_automl_ui ou 03b_automl_sdk, OU\n"
            "  - Cole manualmente o run_id no widget 'run_id'."
        )
    exps_sorted = sorted(
        exps,
        key=lambda e: getattr(e, "last_update_time", 0) or 0,
        reverse=True,
    )
    exp = exps_sorted[0]
    print(f"Experimento mais recente: {exp.name} (id={exp.experiment_id})")

    # Tenta cada métrica primária candidata, da mais comum para a mais
    # genérica. AutoML costuma logar todas; pegamos a primeira que ordena.
    for metric in ("val_f1_score", "f1_score", "val_roc_auc", "roc_auc"):
        runs = client.search_runs(
            [exp.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
        if runs:
            run = runs[0]
            score = run.data.metrics.get(metric)
            print(f"  Melhor run por {metric}={score}: {run.info.run_id}")
            return run.info.run_id

    # Fallback: run mais recente do experimento.
    runs = client.search_runs(
        [exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"O experimento '{exp.name}' não tem nenhum run.\n"
            "Rode antes 03a_automl_ui ou 03b_automl_sdk, OU cole o run_id "
            "manualmente no widget."
        )
    run = runs[0]
    print(f"  Fallback (start_time DESC): {run.info.run_id}")
    return run.info.run_id


if run_id_widget:
    run_id = run_id_widget
    print(f"Usando run_id do widget: {run_id}")
else:
    print("Widget 'run_id' vazio — fazendo busca automática...")
    run_id = _auto_pick_run_id(client, config.user_short)
    print(f"run_id selecionado automaticamente: {run_id}")

# COMMAND ----------
# DBTITLE 1,Registrar a versão do modelo em Unity Catalog

# Cada chamada de register_model cria uma nova versão monotônica do
# modelo `config.model_full_name`. Se for a primeira vez, o objeto modelo
# é criado em UC com a versão 1.
model_uri = f"runs:/{run_id}/model"
print(f"Registrando {model_uri} -> {config.model_full_name}")

mv = mlflow.register_model(
    model_uri=model_uri,
    name=config.model_full_name,
)
print(f"Versão registrada: {mv.version}")

# COMMAND ----------
# DBTITLE 1,Atribuir alias Champion à nova versão

# Aliases em UC são **movíveis**: ao re-atribuir Champion para uma versão
# nova, a anterior automaticamente perde o alias (cada alias aponta para
# exatamente uma versão).
client.set_registered_model_alias(
    name=config.model_full_name,
    alias=ALIAS_CHAMPION,
    version=mv.version,
)
print(
    f"Alias '{ALIAS_CHAMPION}' apontando para a versão {mv.version} "
    f"de {config.model_full_name}."
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Padrão Champion / Challenger
# MAGIC
# MAGIC - **Champion** = versão atualmente servida em produção. O endpoint
# MAGIC   de Model Serving (notebook 06) referencia
# MAGIC   `models:/<modelo>@Champion`, então qualquer movimentação do alias
# MAGIC   reflete no endpoint na próxima atualização de config.
# MAGIC - **Challenger** = candidata em avaliação A/B antes de promover. Pode
# MAGIC   estar sendo testada offline (replay de tráfego) ou online (split de
# MAGIC   tráfego no endpoint, configurando `traffic_config`).
# MAGIC
# MAGIC Fluxo típico de retreino nesta demo:
# MAGIC
# MAGIC 1. Treinar nova versão via `03b_automl_sdk` → run_id novo.
# MAGIC 2. Registrar nova versão (este notebook), mas atribuir `Challenger`
# MAGIC    em vez de `Champion`.
# MAGIC 3. Comparar métricas Champion vs. Challenger no mesmo holdout.
# MAGIC 4. Se Challenger vencer, **mover** `Champion` para a nova versão. O
# MAGIC    alias é único: ao apontar para a nova, a anterior libera o slot.
# MAGIC 5. Atualizar o endpoint (notebook 06) — ele já segue o alias.
# MAGIC
# MAGIC A célula abaixo mostra (comentada) a chamada que promoveria uma
# MAGIC versão `N+1` a `Challenger`. Como esta demo registra uma única
# MAGIC versão, **não executamos** a chamada — é apenas documentação do
# MAGIC padrão.

# COMMAND ----------
# DBTITLE 1,Demonstração do padrão Challenger (não executa)

# Quando houver uma versão N+1 candidata vinda de um novo treino:
#
# next_version = "<numero_da_versao_candidata>"
# client.set_registered_model_alias(
#     name=config.model_full_name,
#     alias=ALIAS_CHALLENGER,
#     version=next_version,
# )
#
# Após validar offline, promover a Champion:
#
# client.set_registered_model_alias(
#     name=config.model_full_name,
#     alias=ALIAS_CHAMPION,
#     version=next_version,
# )
# # opcional: liberar o alias Challenger
# client.delete_registered_model_alias(
#     name=config.model_full_name, alias=ALIAS_CHALLENGER,
# )

print(
    "Demonstração documentada acima. Nesta demo registramos apenas uma "
    "versão, então a célula está intencionalmente comentada."
)

# COMMAND ----------
# DBTITLE 1,Listar versões e aliases atuais do modelo

versions = client.search_model_versions(
    f"name='{config.model_full_name}'"
)

# No UC, search_model_versions não popula aliases nas versões retornadas — é
# preciso chamar get_model_version para cada uma para obter a lista de aliases.
print(f"Versões registradas em {config.model_full_name}:")
print("-" * 72)
for v in sorted(versions, key=lambda x: int(x.version)):
    full = client.get_model_version(
        name=config.model_full_name, version=v.version
    )
    aliases = list(full.aliases) if full.aliases else []
    aliases_str = ", ".join(aliases) if aliases else "-"
    print(
        f"  v{v.version:<3s}  aliases={aliases_str:<30}  run_id={v.run_id}"
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## Modelo registrado e Champion atribuído
# MAGIC
# MAGIC Resumo do que aconteceu:
# MAGIC
# MAGIC - Registramos o melhor run do AutoML como uma nova versão de
# MAGIC   `config.model_full_name` no Unity Catalog Model Registry.
# MAGIC - Atribuímos o alias `Champion` a essa versão. O endpoint de
# MAGIC   Model Serving (notebook 06) vai resolver
# MAGIC   `models:/<modelo>@Champion` para esta versão.
# MAGIC - Documentamos como introduzir um `Challenger` em retreinos futuros.
# MAGIC
# MAGIC **Próximo passo**: rodar `05_inference_notebook/05_inference_notebook`
# MAGIC para consumir o modelo registrado em modo função Python e em modo
# MAGIC Spark UDF (batch).
