# Диссертация
Проект называется PDS - Pretrained Diagnostic System
## Цель диссертации
Целью статьи является создание модели искусственного интеллекта, задача которого диагностировать неисправности колесных пар, а именно ползуны,
неравномерные прокаты и изношенный гребень для повышения безопасности передвижения подвижного состава, снижения ненормативных нагрузок на рельсы, также позволяет эффективнее использовать вагоны на сети
## Общая информация о проекта и средства достижения
Для достижения поставленной цели необходимо собрать базу данных, которая будет основой для обучения модели. Для сбора данных использовался ПК «Универсальный механизм» ("https://github.com/daniilgorenkov/UM-Bot"),
в котором проводилось математическое моделирование полувагона 12-132 на тележках модели 18-100 в различных условиях, а именно:
1.	Порожний и груженый вагоны;
2.	Прямая и кривые 350 м и 650 м;
3.	С исправным колесом, с полузном и неравномерным прокатом;
4.	Скорости от 10 км/ч до 120 км/ч на прямых и до 80 км/ч в кривых с шагом в 
10 км/ч;
5.	С шестью профилям колес:
	 - По ГОСТ 10791 [1];
	 - Гребень 24 мм;
	 - Гребень 26 мм;
	 - Гребень 28 мм;
	 - Гребень 30 мм;
	 - Внутренний профиль колеса в ПК «Универсальный механизм» (newwagonw).
## Структура
|Название|Описание|
|----------|----------|
|observe.ipynb|Основной файл, где проводится обработка результатов и обучение моделей|
|PDS.py|Файл с функционалом проекта|
|stst_results.csv|Статистика работы моделей при различных условиях|
## Инструменты
Pandas, numpy, seaborn, matplotlib, sklearn, XGBoost, Catboost, LightGBM 
