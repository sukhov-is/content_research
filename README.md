# Telegram Research Bot

Telegram-бот для исследования контента каналов: парсинг сообщений, поиск по ключевым словам, классификация через OpenAI API и агрегация статистики с экспортом в CSV.

## Требования

- Python 3.11+
- Telegram Bot Token (от @BotFather)
- Telegram API credentials (api_id, api_hash) с https://my.telegram.org
- OpenAI API key

## Установка

```bash
pip install -r requirements.txt
cp .env.example .env
# Заполнить .env своими данными
```

## Настройка

1. Создайте бота через @BotFather и получите токен
2. Зарегистрируйте приложение на https://my.telegram.org для получения api_id и api_hash
3. Получите OpenAI API key
4. Заполните `.env` файл

## Запуск

```bash
python main.py
```

При первом запуске Telethon запросит код авторизации через SMS.

## Использование

1. Отправьте боту команду `/start`
2. Отправьте YAML-файл конфигурации (см. `config_example.yaml`)
3. Бот начнёт анализ и будет отправлять прогресс
4. По завершении получите статистику и CSV-отчёт

## Конфигурация

См. `config_example.yaml` для примера. Основные параметры:

- `channels` -- список каналов для анализа
- `period` -- временной диапазон (start/end)
- `keywords` -- ключевые слова для поиска
- `classification.categories` -- категории для LLM-классификации
- `classification.tones` -- тона для LLM-классификации
- `llm` -- параметры модели (model, temperature, batch_size, max_concurrent)
