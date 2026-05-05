# Contratos da demo

Especificação canônica que **todos** os notebooks devem respeitar. Mantida em sincronia com `config/demo_config.py`.

---

## Convenção de notebook (`.py` Databricks source)

Cabeçalho obrigatório na primeira linha do arquivo:
```
# Databricks notebook source
```

Marcadores:
- Nova célula: `# COMMAND ----------`
- Markdown: `# MAGIC %md`
- SQL: `# MAGIC %sql`
- `%run`, `%pip`, etc.: `# MAGIC %run ...`
- Título de célula opcional: `# DBTITLE 1,<título>`

Estrutura padrão de cada notebook:

1. Célula `%md` com **objetivo** do notebook (1-2 frases) e contexto (qual notebook anterior, qual posterior).
2. Célula com `%run ../config/demo_config` para carregar contratos.
3. Célula que chama `get_widgets(dbutils, spark)` e `config = resolve_config(dbutils, spark)`.
4. Corpo do notebook usando **sempre** `config.table("nome")`, `config.model_full_name`, `config.endpoint_name`, etc.
5. Célula `%md` final com "**próximo passo**: rodar `0X_<nome>`".

---

## Tabelas (todas em `config.schema_full` = `<catalog>.<schema>`)

Acesso canônico: `config.table("users")`, `config.table("books")`, etc.

| Tabela | Coluna | Tipo | Notas |
|---|---|---|---|
| `users` | `user_id` | STRING | PK |
|  | `signup_date` | DATE | |
|  | `plan_type` | STRING | `free` ou `premium` |
|  | `age_band` | STRING | `<25`, `25-34`, `35-44`, `45-54`, `55+` |
|  | `region` | STRING | regiões fictícias |
|  | `device_type` | STRING | `mobile`, `tablet`, `desktop` |
| `books` | `book_id` | STRING | PK |
|  | `title` | STRING | fictício |
|  | `genre` | STRING | ficção, não-ficção, romance, etc. |
|  | `format` | STRING | `ebook` ou `audiobook` |
|  | `length_minutes` | INT | |
|  | `language` | STRING | `pt`, `en`, `es` |
| `reading_events` | `user_id` | STRING | FK → users |
|  | `book_id` | STRING | FK → books |
|  | `event_ts` | TIMESTAMP | |
|  | `progress_pct` | DOUBLE | 0.0 a 1.0 |
|  | `session_minutes` | INT | |
|  | `finished` | BOOLEAN | |
|  | `rating` | INT | nullable, 1 a 5 |
| `subscriptions` | `user_id` | STRING | FK → users |
|  | `start_date` | DATE | |
|  | `end_date` | DATE | nullable |
|  | `cancel_reason` | STRING | nullable |
| `customer_features` | `user_id` | STRING | PK |
|  | (features) | (várias) | recência, frequência, % conclusão, mix ebook/audio, tenure, etc. |
|  | `will_reactivate` | INT | label binário (0/1) — coluna em `LABEL_COLUMN` |

Volume alvo: `N_USERS = 10_000` usuários, `N_EVENTS_TARGET ≈ 500_000` eventos. Seed `RANDOM_SEED = 42`. Janela: ~24 meses até `snapshot_date`.

---

## Definição do label

`will_reactivate = 1` ⇔ usuário ficou **inativo ≥ `INACTIVITY_WINDOW_DAYS` (30) dias** antes de `config.snapshot_date` E **consumiu algo** (qualquer evento) nos **`PREDICTION_HORIZON_DAYS` (30) dias após** `config.snapshot_date`. Caso contrário 0.

Apenas usuários **elegíveis** (inativos no corte) entram na `customer_features`. Usuários ativos no corte são excluídos do treino.

---

## Modelo

- Nome em UC: `config.model_full_name` → `<catalog>.<schema>.reactivation_model`
- Aliases: `Champion` (em produção), `Challenger` (próxima candidata)
- Registro feito em `04_model_registry` a partir do melhor run do AutoML.

---

## Endpoint Model Serving

- Nome: `config.endpoint_name` (default `automl-reactivation-<user_curto>`)
- Tipo: serverless, **CPU Small**, **scale-to-zero=true**
- **Inference Tables ativadas** com prefix `INFERENCE_TABLE_PREFIX = "inference_log"` no schema do participante
- Model: aliased reference `models:/<catalog>.<schema>.reactivation_model@Champion`

---

## Não fazer

- ❌ Hardcode de catálogo, schema ou nomes de tabela em qualquer notebook. Sempre `config.table(...)` / `config.model_full_name`.
- ❌ `import config.demo_config` — notebooks Databricks não estão num pacote Python. Use `%run ../config/demo_config`.
- ❌ Modificar `config/demo_config.py` no meio do desenvolvimento dos notebooks. Mudanças aqui são contrato global.
- ❌ Mencionar a empresa cliente ou nomes específicos. Apenas "plataforma de audiobooks e ebooks" como contexto genérico.
- ❌ Salvar dados em DBFS ou local. Tudo em Delta dentro de `<catalog>.<schema>`.

## Sempre fazer

- ✅ Primeiro `%run ../config/demo_config`, depois `get_widgets(...)` e `resolve_config(...)`.
- ✅ Comentários e markdown em PT-BR.
- ✅ Mensagens de erro claras quando widgets faltarem ou permissões UC forem insuficientes.
- ✅ Terminar cada notebook com indicação do próximo.
