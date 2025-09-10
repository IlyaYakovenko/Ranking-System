# Ranking-System

Проект по реализации ранжирования (Learning to Rank) на датасете Microsoft LETOR.

## Цель
Продемонстрировать ключевые этапы решения ML-задачи ранжирования: от подготовки данных и вычисления метрик до построения и оценки модели LightGBM Ranker.

## Данные
Используется датасет MQ2008 из коллекции [Microsoft LETOR](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/).

## Метрики
NDCG@5, NDCG@10

## Структура проекта
* `data/` - директория для данных (в git не заливаются)
* `notebooks/` - Jupyter Notebook с исследовательским анализом и обучением модели
* `requirements.txt` - список зависимостей

## Как запустить
1. Клонировать репозиторий
2. Установить зависимости: `pip install -r requirements.txt`
3. Скачать данные и положить в папку `data/`
4. Запустить Jupyter Notebook: `jupyter notebook` и открыть `notebooks/01_ranking_analysis.ipynb`