## Мультимодальная система RAG с использованием ColPali + Qwen-2-VL

![image](https://github.com/user-attachments/assets/89cf16d4-f46e-43e5-9132-6d6a124ac9e7)

Решение задания в рамках соревнования Норникеля. Имеется базовая реализация решения на Streamlit.

Этот репозиторий содержит несколько ноутбуков, реализующих мультимодальную систему на базе **Retrieval-Augmented Generation (RAG)**. Система работает с векторными базами данных и интегрирует текст и изображения для генерации ответов. В частности, используется фреймворк [**byaldi**](https://github.com/AnswerDotAI/byaldi), а модель 🤗 [colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2) предназначена для обработки мультимодальных запросов (текст и изображения). Для обработки текстового запроса с изображениями для инференса используется модель 🤗 [Vikhr-2-VL-2b-Instruct-experimental](https://huggingface.co/Vikhrmodels/Vikhr-2-VL-2b-Instruct-experimental).

Для успешной работы обязательно потребуются данные хакатона, скачать можно [здесь](https://drive.google.com/file/d/1bHGJGOnVtRYCl7LJ8eR7VGwKd6C7eMK6/view?usp=drive_link).

Презентация проекта: [ссылка](https://drive.google.com/file/d/1zIQuq2tAB_irfEUKqE9o7y3hMjgsTv-R/view)

## Информация о ноутбуках

Существует несколько версий решения: 
- [полная версия в одном ноутбуке](https://github.com/metanovus/nornikel-rag-2024/tree/main/notebooks/for_demontration) (выполнение индексации документов через ColPali + генерация ответов через Vikhr-2-VL).
- [две части полной версии](https://github.com/metanovus/nornikel-rag-2024/tree/main/notebooks/for_testing) (Часть 1 - для создания векторной БД, часть 2 - для инференсов).
- [специальная версия](https://drive.google.com/file/d/1OaaVAQY5A9o3QIvGhHMH71Qtlp3m0u0O/view?usp=drive_link) (предназначенная для углубленного тестирования с готовой базой и данными - архив весом около 888 МБ).

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/metanovus/nornikel-rag-2024.git
   cd nornikel-rag-2024
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
   
## Как запустить локально Streamlit
Необходимо выполнить следующие шаги:

1. Убедитесь, что у вас установлен **Docker** и **Docker Compose**.
2. Перейдите в директорию с файлом `docker-compose.yml`.
3. Выполните команду для сборки и запуска контейнера:
   ```bash
   docker-compose up --build
   ```
4. Откройте **Streamlit** в браузере по адресу: `http://localhost:8000`.

Контейнер автоматически запустит Streamlit на порту 8000, как указано в файле `docker-compose.yml`.
