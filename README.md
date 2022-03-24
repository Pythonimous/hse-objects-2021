# hse-objects-2021
## Master's Thesis Repository

### _Thesis_:
#### Image Classification Algorithms based on Object Detection and Structural Pattern Recognition.

## Contents:

- [**data**][dat]: csv files for both tasks
- [**example_photos**][exP]: image examples for demo.ipynb
- [**imports**][imp]: models, data loaders and primary project metrics
- [**ipynb_preproc**][prep]: Kaggle / Colab / local preprocessing files
- [**ipynb_training**][train]: Kaggle / Colab (pytorch) / local (sklearn) training files
- [**pretrained_models**][pretr]: pretrained sklearn logistic regression models for demo.ipynb
- [**scripts**][scr]: download additional files from Google drive (optional):
  - [***download_features.sh***][dF]: download and unpack transfer / object features;
  - [***download_photos.sh***][dP]: download and unpack images.
    > Only images used in at least one of the tasks.
    > The whole dataset is available [here][yelp].
  - [***download_svc.sh***][dS]: Support Vector Classifier download script.
    > For business / restaurant classification only, since logistic regression is not a top method.
- [***config.json***][conf]: Image, Data, Hyperparameter, etc. paths are changed here.
- [***demo.ipynb***][demo]: Illustrated pipeline demo. You can either use the examples from [**example_photos**][exP], or upload your own.

### To install the preprequisites:
```
pip3 install -r requirements.txt
```
### To download and unpack Google Drive images:
```sh
./scripts/download_photos.sh
```
### To download and unzip object features:
```sh
./scripts/download_features.sh
```
### To download the pretrained Support Vector Classifier:
```sh
./scripts/download_svc.sh
```
## Results for classical models on combined transfer + confidence + count vectors:

| | Logistic Regression | NB Classifier | SVC | Случайный лес |
| ------ | ------ | ------ | ------ | ------ |
| **Business / Restaurant** | Acc. 69.98%;<br>F1 0.7130 | Acc. 63.34%;<br>F1 0.6533 | **Acc. 72.72%;<br>F1 0.7446** | Acc. 69.38%;<br>F1 0.7176 |
| **Ambience** | **Jaccard Coef: 0.3582;<br>Hamming Loss: 0.3063;<br>F1: 0.4249;<br>ROC: 0.6742.** | Jaccard Coef: 0.2589;<br>Hamming Loss: 0.4047;<br>F1: 0.3463;<br>ROC: 0.6027. | Дольше 2.5 ч. | Jaccard Coef: 0.0747;<br>Hamming Loss: 0.1832;<br>F1: 0.0657;<br>ROC: 0.5154. |

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
