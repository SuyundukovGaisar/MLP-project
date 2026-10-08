# Многослойный персептрон

## Структура проекта

* `data.csv` — исходный набор данных (Breast Cancer Wisconsin).
* `split.py` — предобработка данных: EDA, заполнение пропусков по среднему класса без утечки данных, Z-score нормализация и стратифицированное разбиение на `train` и `val`.
* `train.py` — создание и обучение архитектуры MLP (поддержка Mini-batch SGD и Adam, интерактивный режим и аргументы CLI).
* `predict.py` — загрузка обученных весов, расчет метрик (Accuracy, BCE, Confusion Matrix, Precision, Recall, F1) и сохранение предсказаний в CSV.
* `report.pdf` — отчет по проекту.

## Инструкция по запуску

### 1. Предобработка и разделение данных
```bash
python split.py --input data.csv
```
Создает файлы `data_train.csv` и `data_val.csv`.

### 2. Обучение модели

**Интерактивный режим** (консоль предложит задать число слоев и нейронов):
```bash
python train.py
```

**Либо запуск через аргументы командной строки:**
```bash
python train.py --layers 64 32 --epochs 100 --batch_size 32 --lr 0.001 --optimizer adam
```
Модель сохраняет веса в `model_weights.npz`.

### 3. Прогнозирование и оценка качества
```bash
python predict.py --model model_weights.npz --data data_val.csv
```
Выводит матрицу ошибок, рассчитывает точность и сохраняет предсказания в `predictions_data_val.csv`.
