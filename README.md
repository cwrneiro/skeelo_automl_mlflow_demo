# Demo Databricks AutoML — Reativação de Clientes

Demonstração standalone de **AutoML, MLflow Model Registry, Model Serving e Inference Tables** no Databricks, em PT-BR. Caso de uso: predizer **reativação de clientes inativos** numa plataforma genérica de audiobooks/ebooks.

A demo é executada num workspace Databricks (você usa seu próprio schema dentro de um catálogo Unity Catalog compartilhado) e cobre, do zero ao deploy:

1. **Setup** do seu schema.
2. **Geração de dados mock** (~10k usuários, ~500k eventos) em Delta.
3. **EDA** e construção da feature table com label.
4. **AutoML** — pela UI e pelo SDK Python.
5. **Registro do modelo** em UC com aliases Champion/Challenger.
6. **Inferência batch** (pyfunc + Spark UDF).
7. **Model Serving** com endpoint serverless e Inference Tables.
8. **Monitoramento** sobre as Inference Tables.
9. **Teardown** do endpoint e (opcional) do schema.

---

## Pré-requisitos

- **Workspace Databricks** com **Unity Catalog** habilitado.
- Acesso a um **catálogo UC** com permissão de `USE CATALOG` + `CREATE SCHEMA`.
- Um **cluster clássico single-user** rodando **DBR 17.3 LTS ML** para os notebooks de geração de dados, EDA e AutoML. Modelos servidos rodam em **serverless** (gerenciado pelo próprio Model Serving).
- Permissão para **criar Model Serving endpoints**.
- (Apenas para o notebook `99_teardown`) permissão de `DROP SCHEMA`.

---

## Como rodar

1. **Clone o repositório** e importe-o como um Databricks Repo (Workspace → Repos → Add Repo).
2. Abra `00_setup/00_setup.py` e ajuste os widgets:
   - `catalog`: catálogo UC onde o schema da demo será criado (**obrigatório**).
   - `schema`: default `automl_demo_<seu_usuário>`. Ajuste se quiser.
   - `endpoint_name`: default `automl-reactivation-<seu_usuário>`.
   - `snapshot_date`: deixe vazio (auto = hoje − 30 dias).
3. Rode os notebooks **na ordem numérica**:

| # | Notebook | Tempo aprox. | Cluster |
|---|---|---|---|
| 0 | `00_setup` | < 1 min | clássico ML |
| 1 | `01_data_generation` | 1-2 min | clássico ML |
| 2 | `02_eda` | 1-2 min | clássico ML |
| 3a | `03_automl_training/03a_automl_ui` | 10-20 min (training pela UI) | clássico ML single-user |
| 3b | `03_automl_training/03b_automl_sdk` | 10-20 min | clássico ML single-user |
| 4 | `04_model_registry` | < 1 min | clássico ML |
| 5 | `05_inference_notebook` | 1-2 min | clássico ML |
| 6 | `06_model_serving` | 10-15 min (provisioning) | clássico ML |
| 7 | `07_monitoring` | < 1 min, **após ~5-10 min de logging** | clássico ML |
| 99 | `99_teardown` | < 1 min | clássico ML |

`03a` e `03b` são equivalentes — você só precisa rodar **um dos dois**. O `03a` é guiado pela UI (ideal pra entender o produto). O `03b` é programático (ideal para CI/CD e reprodutibilidade).

---

## Tipos de instância recomendados (single-node, DBR 17.3 LTS ML)

A demo é leve. Qualquer das opções abaixo serve:

| Cloud | Instance type |
|---|---|
| AWS | `m5d.xlarge` |
| Azure | `Standard_DS3_v2` |
| GCP | `n2-standard-4` |

O endpoint de Model Serving usa workload size **CPU Small**, gerenciado pelo Databricks (você não escolhe instância).

---

## Configuração via widgets

Cada notebook abre com 4 widgets criados por `config/demo_config.py`:

- `catalog` — catálogo UC (obrigatório).
- `schema` — seu schema.
- `endpoint_name` — nome do endpoint Model Serving.
- `snapshot_date` — data de corte para o label (vazio = automático).

**Você só precisa preencher esses widgets uma vez, em `00_setup`.** Ao terminar, `00_setup` grava a configuração em `/Workspace/Users/<seu_email>/.automl_demo_config.json`. Os demais notebooks abrem com esses valores já preenchidos como *default* dos widgets.

Para alterar a configuração global (ex: trocar de catálogo no meio da demo), volte ao `00_setup`, ajuste os widgets e rode-o de novo.

Detalhes do contrato (schemas das tabelas, definição do label, convenções) em [`docs/contracts.md`](docs/contracts.md).
Erros comuns em [`docs/troubleshooting.md`](docs/troubleshooting.md).

---

## Custos

- **Compute clássico**: típico de qualquer notebook em workspace.
- **Model Serving**: cobra por uptime mesmo com `scale-to-zero` (mínimos do plano). **Sempre rode `99_teardown`** ao final.
- **Inference Tables**: armazenamento Delta no seu UC (volume baixo na demo).

---

## Limpeza

**Sempre** rode `99_teardown/99_teardown.py` ao final:

- Deleta o endpoint de Model Serving (sem isso, custo recorrente).
- Opcionalmente dropa o schema (widget `confirm_drop_schema=true`).

Sem o teardown, o endpoint provisionado fica rodando até ser deletado manualmente.

---

## Estrutura do repositório

```
00_setup/                # validação UC, criação do schema
01_data_generation/      # mock data → 4 tabelas Delta
02_eda/                  # exploração + feature table com label
03_automl_training/      # AutoML via UI (03a) e SDK (03b)
04_model_registry/       # registro UC + aliases Champion/Challenger
05_inference_notebook/   # pyfunc.load_model + spark_udf (batch)
06_model_serving/        # endpoint serverless + Inference Tables + REST
07_monitoring/           # queries nas inference tables, drift básico
99_teardown/             # destrói endpoint + (opcional) schema
config/                  # demo_config.py — contratos e helpers
docs/                    # contracts, troubleshooting
```

---

## Licença

Apache 2.0. Veja `LICENSE`.
