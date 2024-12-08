import streamlit as st
import json
import gzip
from PIL import Image, ImageFilter
import shutil
import os
import torch
from vector_index import get_answer_on_query


# Настройки приложения
st.set_page_config(page_title="Норникель RAG", layout="wide")

# Заголовок приложения
st.title("Мультимодальная RAG-система")
st.markdown("### Вопросно-ответная система для Норникель")

# Параметры запроса
query = st.text_input("Введите ваш запрос", "Какой актив формирует наибольшую долю FCF?")
k = st.slider("Количество релевантных документов", min_value=1, max_value=10, value=3)


file_path = '/vector_db/nornikel_index/doc_ids_to_file_names.json.gz'
docs_names = dict()

with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    docs_names = json.load(f)
    docs_names = {int(key): value for key, value in docs_names.items()}


# Инициализация векторной базы и моделей
if st.button("Запустить систему"):
    
    st.info("Обработка запроса...")
    
    answer, images = get_answer_on_query(text_query=query, k=k)

    st.success("Ответы сгенерированы!")
   
    print('Ответ:', answer)

    img1 = Image.open(images)
    img2 = Image.open(images)
    combined_image = Image.concat(images=[img1, img2], direction="horizontal")
    combined_image.show()
    
    print('-' * 50)
   
    shutil.rmtree("'/vector_db/nornikel_index/relevant_img'")
    os.makedirs("'/vector_db/nornikel_index/relevant_img'")

    torch.cuda.empty_cache()

