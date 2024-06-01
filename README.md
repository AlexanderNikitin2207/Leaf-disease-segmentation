# Leaf-disease-segmentation
Нейросетевая модель, сегментирующая участки, поражённые фитофторозом, по мультиспектральным снимкам листьев картофеля. 
Сперва были проведены эксперименты на [наборе данных](https://www.kaggle.com/datasets/fakhrealam9537/leaf-disease-segmentation-dataset), состоящем из цветных снимков растений, поражёнными различными заболеваниями, для выделения наиболее перспективных техник дообучения и нейросетевых архитектур, справляющихся с этой задачей лучше всего. Далее для выделенной модели была проведена процедура fine-tuning на реальных данных, состоящих из мультиспектральных (в видимом и инфракрасном спектрах) снимков листьев картофеля, больных фитофторозом.

## Структура проекта
* rgb image experiments - ноутбуки с экспериментами на найденном наборе данных из цветных снимков
    * prepare_rgd_dataset.ipynb - скачивание данных с платформы Kaggle, проведение train-validation-test разбиения и загрузка полученных наборов данных на платформу HuggingFace
    * fine_tuning_unet.ipynb - эксперименты с различными вариантами дообучения U-Net-подобных архитектур
    * fine_tuning_transformers.ipynb - эксперименты с различными вариантами дообучения transformer-подобных архитектур
* multispectral image experiments - ноутбуки с экспериментами на реальном наборе данных из мультспектральных снимков
    * label_data.ipynb - скрипт для получения сегментационных масок по координатам вершин многоугольников, ограничивающих поражённые области
    * augment_data.ipynb - расширешие набора данных с помощью агументаций
    * Tune_HP.ipynb - подбор опитимальных гиперпараметров модели на расширенном наборе данных
    * Segmormer_tuning_on_multispectral_images.ipynb - дообучение модели с подобранными гиперпараметрами на расширенном наборе данных

## Стек используемых технологий
1. [Python](https://www.python.org/)
2. [Wandb](https://wandb.ai/site)
3. [PyTorch](https://pytorch.org/docs/stable/index.html)
4. [Torchvision](https://pytorch.org/vision/stable/index.html)
5. [🤗 Transformers](https://huggingface.co/docs/transformers/index)
6. [Segmentation models PyTorch](https://smp.readthedocs.io/en/latest/)

## Модель
На основании экспериментов, проведённых на [открытом наборе данных](https://www.kaggle.com/datasets/fakhrealam9537/leaf-disease-segmentation-dataset), была выбрана предобученная версия модели [SegFormer-b2](https://huggingface.co/nvidia/mit-b2).

## Набор данных
Для тестирования метода в реальных условиях был предоставлен набор данных, состоящий из 604 снимков кустов картофеля с разрешением 3456 на 5184 пикселей.. Некоторые из участков были подвергнуты заражению фитофторозом. Датасет содержит снимки листьев картофеля через 6 дней после заражения. 

Изображения были нарезаны на более маленькие разрешением 512 на 512 пикселей. Пример полученных снимков представлен ниже.
![IMG_479](https://github.com/AlexanderNikitin2207/Leaf-disease-segmentation/assets/113451350/4638d717-d2ea-4e8b-8567-552b8f0b5b29 "Изображение в инфракрасном спектре")
![IMG_479](https://github.com/AlexanderNikitin2207/Leaf-disease-segmentation/assets/113451350/ff39f90d-9200-416e-b679-2dbc92952055 "Изображение в видимом спектре")

## Метрики качества
На тестовом множестве модель показала следующие результаты:
* Intersection over Union (IoU): 0.413
* Dice coefficient: 0.584
* Cohen's kappa score: 0.573

 ## Результаты работы модели
 ![result_1](https://github.com/AlexanderNikitin2207/Leaf-disease-segmentation/assets/113451350/a8eb673a-938a-40d0-a789-cb61ecfcd6ac "Результаты работы модели")
 
 ![result_2](https://github.com/AlexanderNikitin2207/Leaf-disease-segmentation/assets/113451350/0a9da9dd-fbb6-49ed-974c-c7ad01e00c04 "Результаты работы модели")

