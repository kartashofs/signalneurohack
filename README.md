# SignalNeuroHack-DigitalDesign-Face-Verification

## Введение
Это - решение для **SignalNeuroHack, 5 кейс, DigitalDesign**

В этом репозитории вы найдете актуальный код для получения высокого качества решения кейса, код для воспроизведения экспериментальных моделей, которые не удалось использовать для получения предсказания на тестовом датасете.


## Решение

### Краткое описание
Код в репозитории поможет со State-Of-The-Art точностью сравнивать схожесть людей на изображениях.
Текущее решение представляет собой имплементацию модели с [IResnet100 Backbone с ArcFaceLoss](https://drive.google.com/file/d/1Gh8C-bwl2B90RDrvKJkXafvZC3q4_H_z/view) и Deep-препроцессингом данных.

### Особенности решения
- Коррекция освещения лица на изображении с помощью [Deep Single Image Portrait Relighting CNN Model](https://zhhoper.github.io/dpr.html)
- Имплементация ArcFace модели выполнена с помощью библиотеки [InsightFace](https://github.com/deepinsight/insightface)


### Внешние данные и Pretrained weights
##### В качестве внешних данных использовались лишь разрешенные.
Мы использовали [Pretrained R100 Model](https://github.com/deepinsight/insightface/tree/master/model_zoo) для инициализации.
Во время тренировки использовался датасет Glint360K - то, что указано в списке предтренированных весов на [странице.](https://github.com/deepinsight/insightface/tree/master/model_zoo)


### Модели
- IResnet100 ArcFace + CNN Relightning
- EfficientNetV2 + Contrastive & Circle Losses

### Финальные результаты
Мы использовали две модели и получили следующие результаты:

1) IResnet100 ArcFace + CNN Relightning
**Public LB: F1 - 0.9739; Recall - 0.9152**
**Private LB: F1 - None; Recall - None**

2) EfficientNetV2 + Contrastive & Circle Losses
**Public LB: не хватило времени на тренировку;**


## Структура репозитория
- **SOTA** - папка, в которой находятся актуальное решение.
  - train.py - train function, cfg

- **experimental** - bonus code (experiments w/ various models)
  EfficientNetCirclenContrastLosses - директория
  -

## Как запустить актуальное решение
- **SOTA** - папка, в которой находятся Python-файлы с необходимыми функциями
  - 

*Don't deal with the noise...*