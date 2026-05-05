# Troubleshooting

Problemas comuns ao rodar a demo, em ordem aproximada de probabilidade.

---

## 1. `Widget 'catalog' obrigatório`

**Sintoma**: `00_setup` (ou qualquer notebook) lança `ValueError: Widget 'catalog' obrigatório`.

**Causa**: o widget `catalog` foi deixado vazio.

**Solução**: preencha o widget `catalog` no topo do notebook com um catálogo UC onde você tenha permissão de `CREATE SCHEMA`. Repita em cada notebook (widgets não cruzam fronteira de notebook).

---

## 2. `Permission denied: CREATE SCHEMA` (ou `CREATE TABLE`)

**Sintoma**: `00_setup` falha no `CREATE SCHEMA` ou no smoke test de `CREATE TABLE`.

**Causa**: seu usuário não tem `USE CATALOG` + `CREATE SCHEMA` no catálogo informado, ou não tem `CREATE TABLE` no schema.

**Solução**:
- Confira no Catalog Explorer que você tem `USE CATALOG` no catálogo.
- Peça ao admin do workspace para conceder `CREATE SCHEMA` no catálogo, ou aponte para um catálogo onde já tem permissão (ex: catálogo de sandbox `users` em alguns workspaces).
- Se já tem o schema mas falta permissão de tabela: `GRANT CREATE TABLE, MODIFY, USE SCHEMA ON SCHEMA <catalog>.<schema> TO <user>`.

---

## 3. AutoML não inicia / falha imediatamente

**Sintoma**: clicar **Start AutoML** na UI ou rodar `automl.classify` retorna erro sobre tipo de cluster.

**Causa**: AutoML **não roda** em compute serverless ou shared. Precisa de cluster clássico **single-user** com runtime ML.

**Solução**: crie um cluster clássico single-user com **DBR 17.3 LTS ML**, anexe ao notebook e tente de novo.

---

## 4. AutoML termina mas sem trials viáveis

**Sintoma**: `automl.classify` retorna sumário com 0 trials bem-sucedidos, ou todas as métricas `NaN`.

**Causas comuns**:
- Label muito desbalanceado (ex: < 50 positivos).
- Coluna de label não é binária.
- Volume insuficiente após filtros.

**Solução**:
- Confira o balanceamento na última célula de `02_eda`. Se a classe positiva tiver menos que ~100 exemplos, gere mais eventos em `01_data_generation` aumentando `N_EVENTS_TARGET`.
- Verifique que `customer_features` tem a coluna `will_reactivate` com valores 0/1.

---

## 5. `04_model_registry` não acha o `run_id`

**Sintoma**: `RuntimeError: nenhum run encontrado para <user_short>`.

**Causa**: a busca automática procura experimentos AutoML pelo seu `user_short`. Se você rodou o `03a_automl_ui` e nomeou o experimento de outra forma, a busca não acha.

**Solução**: cole o `run_id` do best trial diretamente no widget `run_id` de `04_model_registry`. Você pega esse ID na UI do experimento (coluna **Run ID** ao lado do best run) ou no print de `03b_automl_sdk`.

---

## 6. Erro ao registrar modelo: `Registry URI not set` ou `unsupported`

**Sintoma**: `mlflow.register_model(...)` falha com erros sobre o registry workspace-level.

**Causa**: faltou `mlflow.set_registry_uri("databricks-uc")` ou seu workspace não está com UC habilitado para modelos.

**Solução**: o notebook já chama `set_registry_uri`. Se ainda falhar, confirme com o admin que UC está habilitado para modelos (`Model Registry` em UC, não o legado).

---

## 7. Endpoint demora muito para subir (10-15 min)

**Sintoma**: `06_model_serving` parece travado em `create_and_wait`.

**Causa**: provisioning normal de um endpoint novo.

**Solução**: aguarde. Acompanhe pela UI em **Serving → seu endpoint**. Se passar de 20 minutos, abra **Logs** do endpoint na UI — o erro mais comum é `model loading failed`, geralmente por incompatibilidade entre a runtime do modelo e a do serving (problema raro com modelos AutoML).

---

## 8. Chamada REST retorna 401 Unauthorized

**Sintoma**: a célula final de `06_model_serving` falha com 401.

**Causa**: o token capturado do contexto do notebook expirou ou não tem permissão no endpoint.

**Solução**:
- Reanexe o notebook ao cluster e rode de novo.
- Verifique que você é o owner ou tem `Can Query` no endpoint (UI → Serving → Permissions).

---

## 9. Cold start na primeira chamada (~30s)

**Sintoma**: a primeira request ao endpoint demora ~30 segundos; chamadas subsequentes são rápidas.

**Causa**: `scale-to-zero=true` (default da demo). O endpoint hiberna quando ocioso e leva ~30s para acordar.

**Solução**: comportamento esperado. Para eliminar cold start, ajuste o endpoint pela UI desligando `scale-to-zero` (custo aumenta).

---

## 10. `07_monitoring`: tabela de inferência não existe ainda

**Sintoma**: `RuntimeError: Tabela <catalog>.<schema>.inference_log_payload ainda não existe`.

**Causa**: Inference Tables têm latência de logging de **5-10 minutos** após a primeira chamada ao endpoint. A tabela só é criada quando o primeiro batch de logs é flushado.

**Solução**:
- Faça mais 2-3 chamadas ao endpoint (basta rerodar a célula REST de `06_model_serving`).
- Aguarde 5-10 minutos.
- Rode `07_monitoring` de novo.

Se passar de 30 minutos sem a tabela aparecer, confira na UI do endpoint que **Inference Tables** está realmente ligado (Settings → Inference tables → Enabled).

---

## 11. `99_teardown` não consegue dropar schema

**Sintoma**: `Permission denied` no `DROP SCHEMA`.

**Causa**: drop de schema requer `MANAGE` (ou ownership) no schema.

**Solução**:
- Rode com `confirm_drop_schema=false` (default) — apenas o endpoint é deletado, e o schema fica para inspeção posterior.
- Se quiser limpar tudo, peça ao admin para dropar o schema, ou use o Catalog Explorer.

---

## 12. API do AutoML mudou

**Sintoma**: `03b_automl_sdk` falha com `AttributeError` ou `TypeError` em `automl.classify`.

**Causa**: versões muito novas do DBR ML podem mudar a assinatura ou renomear o entrypoint (Mosaic AutoML).

**Solução**:
- Confira a doc oficial: `docs.databricks.com/aws/en/machine-learning/automl/automl-api-reference`.
- Como alternativa, use o `03a_automl_ui` — a UI é estável.
- Adapte o código em `03b_automl_sdk` conforme o método atual e mantenha o restante do fluxo.

---

## 13. Caracteres acentuados quebrados em widgets

**Sintoma**: nomes de schema gerados a partir de e-mails com acentos viram strings esquisitas.

**Causa**: o helper `_short_user` em `config/demo_config.py` faz `lower()` + regex, removendo acentos compostos pode deixar resíduos.

**Solução**: edite o widget `schema` manualmente para um nome ASCII curto (ex: `automl_demo_meu_nome`).
