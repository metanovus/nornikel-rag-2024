# Используем базовый образ Python с поддержкой CUDA
# FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime as base
FROM python:3.9-slim as base


# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    poppler-utils \
    git \  
    && rm -rf /var/lib/apt/lists/*  

# # Установка рабочей директории
WORKDIR /app

# Копируем файл зависимостей
COPY app/requirements.txt requirements.txt  

# Установка pip и зависимостей
RUN pip install --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Установка byaldi из GitHub после установки transformers
RUN pip install git+https://github.com/AnswerDotAI/byaldi.git  # Устанавливаем byaldi из репозитория

# Копируем папку nornikel_index в контейнерФ
COPY ./vector_db/nornikel_index /app/vector_db/nornikel_index/

# Копируем основной файл приложения и другие необходимые файлы
COPY app/main.py /app/main.py
COPY app/vector_index.py /app/vector_index.py
COPY app/vector_index.py /app/vector_index.py
COPY app/for_testing /app/for_testing

# Установка прав на выполнение для скрипта entrypoint
RUN chmod +x /app/entrypoint.sh

# Установка точки входа и команды запуска приложения
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["CMD", "streamlit run main.py --server.port=8000 & jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser"]
