import os
import shutil
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageFilter
from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel

# Загрузка моделей
model_id = "Vikhrmodels/Vikhr-2-VL-2b-Instruct-experimental"
model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="cuda:1")
processor = AutoProcessor.from_pretrained(model_id)
RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", device='cuda:0')

def resize_text(img, max_size=1000):
    '''Изменение размера изображения'''
    width, height = img.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
            
        img = img.resize((new_width, new_height), Image.LANCZOS)
        img = img.filter(ImageFilter.SHARPEN)

    return img


def get_relevant_images_ready(docs_names, results, image_path='/vector_db/nornikel_index/relevant_img'):
    '''Функция-пайплайн для подготовки релевантных страниц к текстовому запросу'''
    images = [docs_names[results[i]['doc_id']] for i in range(len(results))]
    
    image_path=image_path
    
    os.makedirs(image_path, exist_ok=True)
    pages_to_convert = dict()
    
    for doc, page in zip(images, [doc['page_num'] for doc in results]):
        pages_to_convert.setdefault(doc, []).append(page)
    
    ready_images_for_context = []
    
    for i, (pdf, pages) in enumerate(pages_to_convert.items()):
        for j, page in enumerate(pages):
            output_img_name = f'{image_path}/image_{i + 1}_{j + 1}.jpg'
            
            image = convert_from_path(pdf, first_page=page, last_page=page)
            image = resize_text(*image)
            image.save(output_img_name, 'JPEG')

            ready_images_for_context.append(output_img_name)
    
    return ready_images_for_context

def get_answer_on_query(text_query, docs_names, k=3, temperature=0.3):
    '''Функция для поиска по векторной базе и генерации ответа'''
    
    # 1. Поиск релевантных страниц в векторной базе
    results = RAG.search(text_query, k=k)

    # 2. Перевод релевантных страниц в изображения
    relevant_images = get_relevant_images_ready(docs_names, results)

    # 3. Формирование промпта
    content = [
        *[{"type": "image", "image": image} for image in relevant_images],
        {"type": "text", "text": text_query}
    ]

    system_prompt = '''Ты бизнес-ассистент в компании Норникель. По существу и профессионально ответь на вопрос, опираясь на контекст изображений.
    Не пиши, что изображено на картинках.Твоя задача — находить наиболее релевантные результаты для заданного вопроса. Пожалуйста, следуй этим инструкциям:
    1. **Ответ на вопрос**: Если информация, необходимая для ответа на вопрос, содержится в тексте, предоставь детальный и точный ответ, основываясь только на этой информации.
    2. **Отсутствие информации**: Если в тексте нет информации, относящейся к вопросу, ответь: "Я не нашёл информацию в тексте."
    3. **Неуверенность в ответе**: Если ты не уверен в правильности ответа или считаешь, что информация может быть недостаточной, ответь: "Я не уверен в ответе" и предоставьте свой ответ.'''

    few_shot_prompt = '''
    Пример 1:
    Вопрос: Каковы основные преимущества продукта X?
    Ответ: Основные преимущества продукта X включают высокую эффективность, доступность и простоту использования.
    Пример 2:
    Вопрос: Какие шаги необходимо предпринять для запуска проекта Y?
    Ответ: Для запуска проекта Y необходимо выполнить следующие шаги: 1) провести исследование рынка; 2) разработать бизнес-план; 3) собрать команду; 4) запустить маркетинговую кампанию.
    Пример 3:
    Вопрос: Какова история компании Z?
    Ответ: Я не нашёл информацию в тексте.
    Пример 4:
    Вопрос: Какие риски связаны с продуктом A?
    Ответ: Я не уверен в ответе: возможно, стоит уточнить детали о рисках, связанных с продуктом A.
    '''

    
    # Формирование сообщений для модели
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},  # Системный промпт
        {"role": "system", "content": [{"type": "text", "text": few_shot_prompt}]},  # Few-shot примеры
        {"role": "user", "content": content}
    ]
    

    # 4. Токенизация промпта
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    inputs = inputs.to("cuda:1")

    # 5. Генерация ответа
    generated_ids = model.generate(
        **inputs,
        max_length=2048,
        temperature=temperature,
        top_k=100,
        top_p=0.95,
        max_new_tokens=1024
    )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0], relevant_images
