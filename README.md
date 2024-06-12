# Диссертация
Проект называется PDS - Pretrained Diagnostic System
## Цель диссертации
Целью диссертации является создание модели предсказания образования неисправности на поверхности катания
## Структура
|Название|Описание|
|----------|----------|
|observe.ipynb|Основной файл, где проводится обработка результатов и обучение моделей|
|PDS.py|Файл с функционалом проекта|
|stst_results.csv|Статистика работы моделей при различных условиях|
## Инструменты
Pandas, numpy, seaborn, matplotlib, sklearn, XGBoost, Catboost, LightGBM 
## Алгоритм
1. Обрабатываются данные полученные, при динамическом моделировании в ПК "Универсальный механизм" с помощью бота "https://github.com/daniilgorenkov/UM-Bot";
2. Расчеты разделяются на под расчеты по принципу полного оборота колеса;
3. Для каждого под расчета вычисляются признаки, которые в дальнейшем образуют матрицу признаков;
4. После производится обучение моделей.
