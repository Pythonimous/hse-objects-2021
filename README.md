# hse-objects-2021
## Материалы выпускной квалификационной работы Николаева Кирилла, 19ИАД.

### _Тема выпускной квалификационной работы_:
#### Алгоритмы классификации изображений на основе нейросетевых детекторов объектов и структурного распознавания образов.

## Содержимое проекта:

- [**data**][dat]: csv файлы для обеих задач
- [**example_photos**][exP]: примеры изображений для demo.ipynb
- [**imports**][imp]: модели, загрузчики данных и основные метрики проекта
- [**ipynb_preproc**][prep]: файлы, использовавшиеся для предобработки в Kaggle / Colab / локально
- [**ipynb_training**][train]: файлы, использовавшиеся для обучения в Kaggle / Colab (pytorch) / локально (sklearn)
- [**pretrained_models**][pretr]: предобученные sklearn модели логистической регрессии для demo.ipynb
- [**scripts**][scr]: скрипты для загрузки доп файлов с Google Drive (опционально):
  - [***download_features.sh***][dF]: скрипт для загрузки и распаковки transfer / объектных признаков;
  - [***download_photos.sh***][dP]: скрипт для загрузки и распаковки изображений.
    > Только изображения, встречающиеся хотя бы в одной из задач.
    > Весь датасет можно скачать [здесь][yelp].
  - [***download_svc.sh***][dS]: скрипт для загрузки Support Vector Classifier.
    > Для business / restaurant классификации, поскольку логистическая регрессия не топ-метод.
- [***config.json***][conf]: все конфигурации, собранные в одном месте. пути к изображениям, датасетам, гиперпараметры и проч. изменяются здесь.
- [***demo.ipynb***][demo]: иллюстрированная демонстрация пайплайна со всеми этапами. Можно как использовать примеры из [**example_photos**][exP], так и загружать свои.

### Для установки необходимых python-библиотек:
```
pip3 install -r requirements.txt
```
### Для загрузки и распаковки изображений с Google Drive:
```sh
./scripts/download_photos.sh
```
### Для загрузки и распаковки transfer / объектных признаков с Google Drive:
```sh
./scripts/download_features.sh
```
### Для загрузки предобученного Support Vector Classifier с Google Drive:
```sh
./scripts/download_svc.sh
```
##### Результаты для топ классификаторов на комбинированных векторах transfer + confidence + count:

| | Логистическая регрессия | SVC |
| ------ | ------ | ------ |
| **Business / Restaurant** | Acc. 69.98%;  F1 0.7130 | **Acc. 72.72%; F1 0.7446** |
| **Ambience** | **Jaccard Coef: 0.3582; Hamming Loss: 0.3063; F1: 0.4249; ROC: 0.6742.** | Дольше 2.5 ч. |

   [yelp]: <https://www.yelp.com/dataset/>
    
   [dat]: <https://github.com/Pythonimous/hse-objects-2021/tree/main/data>
   [exP]: <https://github.com/Pythonimous/hse-objects-2021/tree/main/example_photos>
   [imp]: <https://github.com/Pythonimous/hse-objects-2021/tree/main/imports>
   [prep]: <https://github.com/Pythonimous/hse-objects-2021/tree/main/ipynb_preproc>
   [train]: <https://github.com/Pythonimous/hse-objects-2021/tree/main/ipynb_training>
   [pretr]: <https://github.com/Pythonimous/hse-objects-2021/tree/main/pretrained_models>
   [scr]: <https://github.com/Pythonimous/hse-objects-2021/tree/main/scripts>
   
   [dF]: <https://github.com/Pythonimous/hse-objects-2021/blob/main/scripts/download_features.sh>
   [dP]: <https://github.com/Pythonimous/hse-objects-2021/blob/main/scripts/download_photos.sh>
   [dS]: <https://github.com/Pythonimous/hse-objects-2021/blob/main/scripts/download_svc.sh>
   [conf]: <https://github.com/Pythonimous/hse-objects-2021/blob/main/config.json>
   [demo]: <https://github.com/Pythonimous/hse-objects-2021/blob/main/demo.ipynb>
