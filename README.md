## Задача

- Тип задачи: Классификация
- Датасет:
- Метрика качества:

Краткое описание задачи (1–3 предложений).

## Результаты

| Метрика | Train | Validation |
|     accuracy    |       |            |
|     recall    |       |            |
|     precision    |       |           |
|     f1-score    |       |            |

## Подготовка датасета

Описание шагов, как скачать датасет и подготовить его для дальнейших обучения и валидации.

## Как воспроизвести

Команда обучения:
python train.py
Команда валидации:
python val.py
## Development

- Установите [uv](https://docs.astral.sh/uv/getting-started/installation/) и синхронизируйте зависимости:

```bash
uv sync --frozen
```

- Создание и активация виртуального окружения:

```bash
python -m venv .venv
source .venv/bin/activate
```
