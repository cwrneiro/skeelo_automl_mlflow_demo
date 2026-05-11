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
# MAGIC No Databricks atual, o produto se chama **Mosaic AutoML**, e o fluxo
# MAGIC pela UI é iniciado em **Experiments → Classification** (já entra com
# MAGIC o tipo de problema pré-selecionado).

# COMMAND ----------
# MAGIC %md
# MAGIC ## Passo 1 — Abrir o formulário de AutoML
# MAGIC
# MAGIC Na barra lateral do workspace, vá em **Experiments** (1) e clique no
# MAGIC tile **Classification** (2). O tipo de problema (`Classification`) já
# MAGIC fica selecionado pelo próprio ponto de entrada, então você cai direto
# MAGIC no formulário com seções: *Compute*, *Dataset*, *Prediction target* e
# MAGIC *Advanced configuration*.
# MAGIC
# MAGIC ![Experiments → Classification](../docs/img/03a_step1_experiments_classification.png)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Passo 2 — Configurar o experimento
# MAGIC
# MAGIC Você cai na tela **Configure Classification Experiment**. Os 5 campos
# MAGIC numerados na imagem são todos preenchidos aqui — os valores exatos já
# MAGIC foram impressos na célula "Imprimir os nomes que você vai usar na UI"
# MAGIC lá em cima.
# MAGIC
# MAGIC ![Configure Classification Experiment](../docs/img/03a_step2_configure_form.png)
# MAGIC
# MAGIC 1. **Cluster** — selecione um cluster **clássico single-user** com
# MAGIC    **DBR 17.3 LTS ML**. AutoML **não roda** em compute serverless nem
# MAGIC    em clusters *shared / no-isolation*. Se não tiver um compatível,
# MAGIC    crie antes de seguir.
# MAGIC 2. **Input training dataset** — clique em *Browse* e selecione a
# MAGIC    tabela `customer_features` no seu schema. O nome completo é o
# MAGIC    impresso acima como `Dataset (UC table)`.
# MAGIC 3. **Prediction target** — selecione `will_reactivate` (impresso como
# MAGIC    `Prediction target`).
# MAGIC 4. **Experiment name** — pode deixar o default, mas a sugestão
# MAGIC    impressa acima inclui seu `user_short` e facilita encontrar o
# MAGIC    experimento depois.
# MAGIC 5. **Schema (painel da direita)** — desmarque o checkbox **Include**
# MAGIC    da coluna `user_id`. É a chave primária da feature table, não tem
# MAGIC    sinal preditivo, e a cardinalidade altíssima degradaria os modelos.
# MAGIC    (Esse painel substituiu o antigo campo "Excluded columns" do
# MAGIC    Advanced Configuration.)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Passo 3 — Advanced Configuration e iniciar
# MAGIC
# MAGIC Continuando a mesma tela, agora cobrimos os passos numerados 6 a 9 da
# MAGIC imagem (a numeração segue de onde o Passo 2 parou).
# MAGIC
# MAGIC ![Advanced Configuration e Start AutoML](../docs/img/03a_step3_advanced_config_and_start.png)
# MAGIC
# MAGIC 6. **Advanced Configuration (optional)** — clique no header pra
# MAGIC    expandir a seção. Vem recolhida por padrão.
# MAGIC 7. **Evaluation metric** — já vem `F1 score` por padrão, que é
# MAGIC    adequada para o nosso label binário desbalanceado. Você pode
# MAGIC    trocar por `ROC AUC` se preferir; **evite `accuracy`** — com
# MAGIC    desbalanceamento de classes ela engana.
# MAGIC 8. **Timeout (minutes)** — coloque `15`. Budget maior produz modelos
# MAGIC    melhores mas demora mais; `10–15` é suficiente pra demo.
# MAGIC 9. **Start AutoML** — clique no botão no canto inferior esquerdo. O
# MAGIC    formulário fecha e você é redirecionado para a página de progresso
# MAGIC    do experimento.
# MAGIC
# MAGIC Os demais campos (Experiment directory, Training frameworks,
# MAGIC Positive label, Time column, Intermediate data storage) ficam com os
# MAGIC valores padrão. Ao clicar em Start AutoML, o formulário fecha e você é
# MAGIC redirecionado para a página de progresso do experimento — coberta no
# MAGIC próximo passo.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Passo 4 — Acompanhar o experimento rodando
# MAGIC
# MAGIC Logo após o Start AutoML, você cai na página de progresso do
# MAGIC experimento. Ela mostra, em tempo real, os trials sendo criados e
# MAGIC avaliados.
# MAGIC
# MAGIC ![Experimento em execução](../docs/img/03a_step4_training_in_progress.png)
# MAGIC
# MAGIC Pontos que valem atenção nessa tela:
# MAGIC
# MAGIC - **Progresso (Configure → Train → Evaluate)**: a etapa *Train* fica
# MAGIC   ativa enquanto o budget de timeout não esgota.
# MAGIC - **Overview** confirma o que você configurou: training dataset,
# MAGIC   target column, evaluation metric (`val_f1_score`) e timeout.
# MAGIC - **Stop experiment**: dá pra interromper antes do timeout se já
# MAGIC   tiver um resultado bom o bastante.
# MAGIC - **Lista de runs (parte inferior)**: cada linha é um trial do
# MAGIC   AutoML — `lightgbm`, `xgboost`, `sklearn` com diferentes
# MAGIC   hiperparâmetros. À medida que os trials terminam, a coluna de
# MAGIC   métrica (`test_f1_score`) vai sendo preenchida. Use o sort por essa
# MAGIC   coluna para ver o líder corrente.
# MAGIC - O run **`Training Data Storage and Exploration`** aparece logo no
# MAGIC   início — é o notebook de EDA automática gerado pelo AutoML, com
# MAGIC   estatísticas, missing e correlações da feature table.
# MAGIC
# MAGIC Você não precisa ficar parado olhando: a página atualiza sozinha. Vá
# MAGIC para o **Passo 5** quando a etapa *Evaluate* ficar ativa (ou quando
# MAGIC você der Stop).

# COMMAND ----------
# MAGIC %md
# MAGIC ## Passo 5 — Localizar o resultado
# MAGIC
# MAGIC Quando o experimento termina, o progresso fica **Configure ✓ → Train
# MAGIC ✓ → Evaluate ✓** e o Overview mostra **AutoML Evaluation: complete**.
# MAGIC
# MAGIC ![Experimento finalizado](../docs/img/03a_step5_experiment_complete.png)
# MAGIC
# MAGIC Dois elementos importam aqui:
# MAGIC
# MAGIC - **Melhor modelo** — o painel *Model with best val_f1_score* tem o
# MAGIC   botão **View notebook for best model**. Esse notebook é gerado
# MAGIC   automaticamente, contém o código exato do trial vencedor (features,
# MAGIC   pré-processamento, hiperparâmetros, fit) e é totalmente
# MAGIC   reproduzível / editável. Vale dar uma olhada como referência.
# MAGIC - **Runs** — a tabela na parte inferior lista todos os trials. Você
# MAGIC   pode ordenar por `val_f1_score` ou `test_f1_score` para ver o
# MAGIC   ranking e clicar em qualquer run para inspecionar métricas,
# MAGIC   parâmetros e artefatos.
# MAGIC
# MAGIC ### Pegar o run_id do best trial
# MAGIC
# MAGIC O próximo notebook (`04_model_registry`) precisa do `run_id` do best
# MAGIC trial para registrar o modelo em UC. Você tem duas opções:
# MAGIC
# MAGIC 1. **Manual** — clique no melhor run da tabela (linha do topo após
# MAGIC    ordenar por `val_f1_score`) e copie o `Run ID` do canto superior
# MAGIC    direito. Cole no widget `run_id` do `04_model_registry`.
# MAGIC 2. **Automático** — deixe o widget `run_id` do `04_model_registry`
# MAGIC    vazio: o notebook busca sozinho o experimento mais recente cujo
# MAGIC    nome contém seu `user_short` e seleciona o melhor run por
# MAGIC    `val_f1_score`.

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
