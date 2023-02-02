# TweetsAnalyzer
## 1. Описание проекта: 
Твиттер стал важным каналом связи во время чрезвычайных ситуаций.
Повсеместное распространение смартфонов позволяет людям сообщать о чрезвычайной ситуации, которую они наблюдают, в режиме реального времени. Из-за этого все больше агентств заинтересованы в программном мониторинге Twitter (например, организации по оказанию помощи при стихийных бедствиях и информационные агентства).

Но не всегда ясно, действительно ли слова человека предвещают катастрофу. Данный сервис предназначен для классификации таких сообщений на 2 класса : `{'Твит о реальном бедствтии': 'real tweet', 'Твит о фейковом бедствии': 'fake tweet'}`


## 2. Структура проекта 
```
├── LICENSE
├── README.md                      <- The top-level README for developers using this project.
├── data
│   ├── df_for_plot_on_map.csv     <- Data for train model
│   ├── train.csv                  <- Data for train model
│   └── test.csv                   <- Data for test model
│   
├── app
│   ├── log_utils.py               <- Module for set app logger
│   ├── model_park.py              <- Module with ml-models and data handler
│   └── web_server.py              <- FastAPI web-server
│   
├── models                         <- Models storage dir
│   
├── notebooks
│   ├── ModelResearch.html         <- Data analysis and model researsh
│   └── ModelResearch.ipynb        <- Data analysis and model researsh notebook
│   
├── utility_scripts
│   ├── simple_client.py           <- Web-server API simple client
│   └── UI.py                      
│       ├── plot_tools.py          <- Data preparing script for GEO-MAP
│       └── streamlit_ui.py        <- Streamlit UI
│   
└── Test_task.html                 <- Project task
```

## 3. Запуск проекта
Для запуска проекта достаточно скачать предобученную модель и положить в директорию `models` (или обучить заново см. п. 4), а также запустить веб-сервер

```bash
# Запуск веб-сервера
$ cd app
$ python web_server.py
```

Пример вызова веб-сервиса на python:

```python
import requests

URL = "http://127.0.0.1:5000/analyze_tweet/"
DATA ={
    'keyword': ['aaa'], 
    'location': ['NewsYork'], 
    'text': ["Our Deeds are the Reason of this #earthquake M"]
}

resp = requests.post(URL, json={**DATA})
print(resp.text)
```

У сервиса так же есть встроенная документация. Доступ предоставляется через Swagger UI: http://127.0.0.1:5000/docs/

## 4. Обучение ML-модели
**Отчёт по анализу данных и поиску модели можно найти [тут](https://htmlpreview.github.io/?https://github.com/tedey-01/TweetsAnalyzer/blob/master/notebooks/ModelResearch.html)**
Для переобучения модели достаточно либо воспроизвести `ModelResearch.ipynb`, липо выполнить следующую последовательность команд: 

```bash 
$ cd ./app
# uncomment code for train model in `__main__` namespace
$ python model_park.py
```

## 5. Метрики качества 
На текущий момент качество модели LogReg составило:

|    metric    | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
|      0       |    0.83   |  0.81  |   0.82   |   1309  |
|      1       |    0.75   |  0.77  |   0.76   |   959   |
|              |           |        |          |         |
|   accuracy   |           |        |   0.80   |   2268  |
|  macro avg   |    0.79   |  0.79  |   0.79   |   2268  |
| weighted avg |    0.80   |  0.80  |   0.80   |   2268  |

## 6. User Interface

К сервису написан небольшой UI на Streamlit. Общение между ними происходит через Http запросы. 
Для запуска необходимо выполнить следующие команды: 

```bash 
$ cd utility_scripts/UI
$ streamlit run streamlit_ui.py
```
User Interface выглядит следующим образом: 

