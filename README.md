# RAG-прототип: поиск по нормативной электротехнической документации

Локальный RAG-контур для поиска и ответов на вопросы по русскоязычной нормативной документации в электроэнергетике (ФЗ-35, постановления Правительства РФ, приказы Минэнерго). Релевантные фрагменты находятся гибридным поиском в Qdrant, развёрнутый ответ со ссылками на документы генерирует локальная LLM через Ollama.

---

## Содержание

- [Описание](#описание)
- [Структура проекта](#структура-проекта)
- [Установка](#установка)
- [Использование](#использование)
- [Переменные окружения](#переменные-окружения)
- [Contributing](#contributing)
- [Лицензия](#лицензия)

---

**Индексирование (офлайн).** PDF/DOCX/TXT-документы из `data/raw/` очищаются от артефактов (колонтитулы КонсультантПлюс, нумерация страниц), сканы распознаются через Tesseract OCR, таблицы сериализуются в markdown на своей позиции в тексте. Тексты режутся на чанки по границам статей и пунктов: номер пункта попадает в метаданные, длинные пункты дробятся по 800 символов с перекрытием 150, таблицы — только по границам строк с дублированием шапки. Чанки индексируются в Qdrant эмбеддинг-моделью `deepvk/USER-bge-m3`.

**Запрос (онлайн).** Вопрос обрабатывается гибридным поиском — семантическим (dense) и лексическим BM25 (sparse, русский стемминг) параллельно, результаты сливаются в Qdrant через Reciprocal Rank Fusion. По top-k чанкам LLM генерирует ответ строго по найденному контексту. Список использованных документов добавляется к ответу программно из метаданных, служебные теги моделей (`<think>` и т.п.) вырезаются пост-обработкой.

| Компонент       | Технология                                          |
|-----------------|-----------------------------------------------------|
| Оркестрация     | LangChain (LCEL)                                    |
| Векторная БД    | Qdrant (Docker, порт 6333)                          |
| Поиск           | Гибридный: dense + sparse BM25 (`Qdrant/bm25`), RRF |
| Эмбеддинги      | `deepvk/USER-bge-m3` (HuggingFace, локально)        |
| LLM             | Ollama, модель `qwen3:4b` (порт 11434)              |
| API-сервер      | FastAPI, OpenAI-совместимый endpoint (порт 8000)    |
| Веб-интерфейс   | Open WebUI (Docker, порт 3000)                      |
| Оценка качества | RAGAS (faithfulness, answer_relevancy и др.)        |

---

## Структура проекта

```
RAG_prototype/
├── .env                      # переменные окружения (Qdrant, Ollama, эмбеддинги)
├── README.md
├── docker-compose.yml        # Open WebUI (Qdrant запускается отдельным контейнером)
├── data/
│   ├── raw/                  # исходные документы (PDF, DOCX, TXT)
│   └── processed/            # очищенные тексты (один .txt на документ)
└── src/
    ├── ingestion.py          # загрузка, очистка, OCR сканов, чанкинг
    ├── indexing.py           # эмбеддинги + загрузка в Qdrant
    ├── retrieval.py          # гибридный поиск
    ├── generation.py         # RAG-цепочка (промпт, очистка вывода, источники)
    ├── evaluation.py         # оценка качества через RAGAS
    └── api.py                # OpenAI-совместимый сервер поверх RAG-цепочки
```

---

## Установка

### Требования

- Python 3.11+
- Docker (для Qdrant и Open WebUI)
- [Ollama](https://ollama.com/) с моделью `qwen3:4b`
- Tesseract OCR с русским языковым пакетом (для сканированных PDF):
  - macOS: `brew install tesseract tesseract-lang`
  - Linux (Debian/Ubuntu): `sudo apt install tesseract-ocr tesseract-ocr-rus`
  - Windows: [установщик UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki), при установке отметить русский язык; путь к `tesseract.exe` добавить в `PATH`

### Шаги

```bash
git clone <repo-url>
cd RAG_prototype

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install \
  langchain==0.3.* langchain-community==0.3.* langchain-huggingface==0.1.* \
  langchain-qdrant==0.2.* qdrant-client==1.12.* fastembed langchain-openai \
  sentence-transformers==3.3.* transformers==4.47.* torch==2.5.* \
  pypdf==5.* pymupdf==1.28.* pytesseract pillow python-docx \
  python-dotenv==1.0.* openai==1.57.* ragas==0.2.* \
  fastapi uvicorn
```

### Запуск Qdrant

```bash
docker run -d \
  --name qdrant_local \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:v1.12.0

curl http://localhost:6333/healthz   # проверка
```

В PowerShell вместо `$(pwd)` использовать `${PWD}` (и убрать `\`-переносы строк или заменить их на `` ` ``).

### Настройка Ollama

macOS:

```bash
brew install ollama
brew services start ollama          # порт 11434, автозапуск при логине
```

На macOS Ollama ставится нативно (не в Docker) — только так доступна Metal-акселерация Apple Silicon.

Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh   # ставит и запускает systemd-сервис ollama
```

Windows: скачать установщик с [ollama.com/download](https://ollama.com/download) и запустить — сервис стартует автоматически и висит в трее. На машинах с NVIDIA GPU (Linux/Windows) Ollama использует CUDA автоматически, ничего настраивать не нужно.

Дальше на любой платформе:

```bash
ollama pull qwen3:4b                      # модель контура (~2.6 ГБ)
curl http://localhost:11434/api/version   # проверка сервиса
```

### Индексирование документов

Выполняется один раз и при обновлении корпуса. Здесь и далее команды даны для macOS/Linux; на Windows вместо `.venv/bin/python3` — `.venv\Scripts\python`, вместо `.venv/bin/uvicorn` — `.venv\Scripts\uvicorn`.

```bash
# 1. Положить исходные документы в data/raw/
# 2. Предобработка: загрузка, очистка, OCR -> data/processed/
.venv/bin/python3 -m src.ingestion
# 3. Чанкинг + эмбеддинги + загрузка в Qdrant
.venv/bin/python3 -m src.indexing
```

---

## Использование

### Вариант 1: из Python

Требуются запущенные Qdrant и Ollama.

```bash
# разовый вопрос (задаётся в __main__ файла src/generation.py)
.venv/bin/python3 -m src.generation

# только поиск, без генерации
.venv/bin/python3 -m src.retrieval
```

Из своего кода:

```python
from src.generation import build_rag_chain, stream_answer

chain = build_rag_chain(top_k=10)
print(chain.invoke("Что такое технологическое присоединение?"))

# либо со стримингом в консоль
stream_answer("Что такое технологическое присоединение?")
```

Ответ завершается блоком «Использованные нормативные документы», который формируется из метаданных Qdrant.

### Вариант 2: Open WebUI (чат в браузере)

Open WebUI подключается не к LLM напрямую, а к `src/api.py` — OpenAI-совместимому серверу поверх полной RAG-цепочки. В чате это выглядит как одна «модель» `rag-normative-docs`; каждый запрос проходит путь: поиск в Qdrant -> генерация в Ollama -> очистка вывода -> добавление источников. Адрес API передаётся контейнеру через `OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1`, поэтому сервер должен слушать `0.0.0.0`.

```bash
docker start qdrant_local                                    # 1. Qdrant
brew services start ollama                                   # 2. Ollama, если не запущен (Linux/Windows — сервис уже стартует сам)
.venv/bin/uvicorn src.api:app --host 0.0.0.0 --port 8000     # 3. RAG API-сервер
docker compose up -d                                         # 4. Open WebUI
```

Открыть `http://localhost:3000` и выбрать модель `rag-normative-docs`. Первые токены приходят с задержкой: qwen3 сначала «думает» в скрытом блоке `<think>`, который фильтруется сервером.

Проверка API без браузера:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "rag-normative-docs",
       "messages": [{"role": "user", "content": "Что такое технологическое присоединение?"}]}'
```

В `docker-compose.yml` для локальной разработки выключена аутентификация (`WEBUI_AUTH=False`); для продакшна её нужно включить и задать `WEBUI_SECRET_KEY`.

---

## Переменные окружения

Файл `.env` в корне проекта:

| Переменная               | Значение по умолчанию    | Описание                                |
|--------------------------|--------------------------|-----------------------------------------|
| `QDRANT_HOST`            | `localhost`              | Хост Qdrant                             |
| `QDRANT_PORT`            | `6333`                   | HTTP-порт Qdrant                        |
| `QDRANT_COLLECTION_NAME` | `normative_docs`         | Имя коллекции                           |
| `OLLAMA_BASE_URL`        | `http://localhost:11434` | Нативный API Ollama (клиент ChatOllama) |
| `LLM_MODEL`              | `qwen3:4b`               | Тег модели в Ollama                     |
| `OLLAMA_NUM_CTX`         | `16384`                  | Контекстное окно LLM (`num_ctx`)        |
| `EMBEDDING_MODEL_NAME`   | `deepvk/USER-bge-m3`     | Эмбеддинг-модель HuggingFace            |
| `EMBEDDING_DEVICE`       | `mps`                    | `mps` (Apple Silicon), `cuda` (NVIDIA) или `cpu` |

---

## Contributing

1. Сделайте fork репозитория.
2. Создайте ветку для вашей задачи: `git checkout -b feature/my-feature`
3. Внесите изменения и добавьте тесты при необходимости.
4. Зафиксируйте изменения: `git commit -m 'feat: add my feature'`
5. Отправьте ветку: `git push origin feature/my-feature`
6. Откройте Pull Request.

---

## Лицензия

MIT
