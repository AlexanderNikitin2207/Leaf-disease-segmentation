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
