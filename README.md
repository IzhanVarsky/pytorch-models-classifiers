# pytorch-models-classifiers

[//]: # (## VisionTransformer)

[//]: # ()

[//]: # (### Архитектура)

[//]: # ()

[//]: # (* Принимает на вход изображения `224 x 224 x 3`)

[//]: # (* Разбиваем изображение на патчи размером `16 x 16` &#40;всего получается `14 x 14` патчей&#41;)

[//]: # (* Вектор этих патчей подается на вход модели)

[//]: # (* Полносвязный слой `nn.Linear&#40;channels * patch_size * patch_stride, d_model&#41;` переводит во внутреннюю)

[//]: # (  размерность `d_model` &#40;по дефолту равна `768`&#41;)

[//]: # (* )

## InceptionV3:

* [InceptionV1 paper](https://arxiv.org/pdf/1409.4842v1.pdf)
* [Inception V2 and V3 paper](https://arxiv.org/pdf/1512.00567v3.pdf)

### Общие особенности сети Inception:

* Принимает на вход изображения `299 х 299 x 3`
* Имеет `AuxClassifier(-s)` - дополнительный выход сети, который ведет себя как регуляризатор и позволяет бороться с
  затуханием градиентов
* Прием inception: вход обрабатывается несколькими функциями, а затем их выходы конкатенируются, образуя
  общий выход
* Свертки с ядром `1 x 1` - позволяют уменьшить/увеличить число фильтров, не изменяя высоту и ширину изображения
* Смысл сверток `1 x 1`:
    * Допустим, есть свертка `C = Conv2d(in_channels=F1, out_channels=F2, kernel_size=(4, 4))`
    * Вместо того чтобы сразу её выполнять, можно с помощью `1 x 1` свертки вначале произвести уменьшение из `F1` в
      меньшее
      число каналов `X` и затем применить исходную свертку `C` (но с меньшим числом входных каналов - `X`)
    * Почему это выгоднее: пусть `F1 = 256, F2 = 256, X = 64`, тогда в первом случае
      будет `F1 * F2 * 4 * 4 = 256 * 256 * 4 * 4 = 1048576` весов в свертке, а во втором случае
      будет `F1 * X * 1 * 1 + X * F2 * 4 * 4 = 256 * 64 * 1 * 1 + 64 * 256 * 4 * 4 = 278528`, что в `3.7` раза меньше!
    * Почему не ухудшается качество: ожидается, что в активациях рядом лежащие элементы сильно скореллированы, поэтому
      перед использованием свертки было бы разумно сжать их, что и делает свертка `1 x 1` (сжимая фильтры)

### Основные нововведения InceptionV3 относительно предыдущих версий:

* Вместо того чтобы использовать свертку `5 х 5`, её можно заменить на две последовательных свертки `3 х 3`, что
  вычислительно выгоднее (т.к. `f1 * f2 * 5 * 5 > (f1 * f2 * 3 * 3) * 2`, что дает выигрыш в `28%`)
* Аналогично свертка `7 x 7` заменяется на три свертки `3 x 3`
* Развивая дальше эти идеи, предлагается заменить свертки `n x n` на последовательность из двух сверток `1 x n`
  и `n x 1`
* Для одновременного уменьшения размерности изображения и увеличения количества каналов также используется `Inception`,
  состоящий из двух веток `conv` и одной ветки `pool`
* Добавлен `BatchNorm` к AuxClassifier
* Был введен метод регуляризации `Label Smoothing` - сглаживание target меток (`smoothing=0.1`)
* Оптимизатор `RMSProp` вместо ранее использованных `SGD` и `Momentum`

### Приколы и проблемы статьи

Я нигде не смог найти код, полностью соответствующий статье.

Проблемы самой статьи:

* Непонятно, что есть `inceptionV2`, а что `inceptionV3`
* Нигде не указано внутреннее число каналов у `inception` модулей
* Вскользь упомянуто про использование `grid reduction` между блоками `inception`, но не ясно, являются ли они частью
  этих `inception` модулей или же это отдельные блоки
* Не указано, как выглядит блок `AuxClassifier`
* Много опечаток и есть неверная ссылка на картинки (в одном месте написано про фигуру 6, а в другом про фигуру 5, хотя
  явно имелась в виду фигура 6)
* Фигура 10 содержит два разных подхода к `grid reduction`, в описании сказано, что они аналогичны друг другу, но какой
  именно использовался в реализации - не указано

В `PyTorch` официальная модель `InceptionV3` (перенесенная с официального кода на `TensorFlow`):

* В блоке `stem`:
    * использует два `maxpool'инга` вместо написанного в статье одного
    * после первого `maxpool` использует две свертки `1 x 1` и `3 x 3` вместо трех сверток `3 x 3`
* В блоке `inceptionX3` постепенно увеличивает число каналов, хотя в статье сказано, что везде число каналов равно `288`
* В статье (мутно) сказано, что между блоками `inception` нужно использовать `grid reduction`, в
  PyTorch `grid reduction` уже является частью `inception` блоков
* PyTorch во втором блоке `grid reduction` использует свертки `1 x 7` и `7 x 1`, хотя в статье было только про `3 x 3`
  в `grid reduction`

Другие реализации из интернета также имеют свои особенности и отличия от того, что написано в статье...

Поэтому моя реализация - это переделанный код из PyTorch, максимально приближенный к подходу, описанному в статье

### Датасеты, логи, результаты

Модели были обучены и протестированы на двух датасетах:

* [fruits](https://www.kaggle.com/datasets/moltean/fruits)
* [cars](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder)

Итоги:

Лучшие F1_macro на тесте на датасете Cars:

|         |   InceptionV3<br/>PyTorch pretrained    | InceptionV3<br/>PyTorch not pretrained  |             My InceptionV3              |
|:-------:|:---------------------------------------:|:---------------------------------------:|:---------------------------------------:|
| RMSProp |            0.2936 (87 эпоха)            |  не более 0.0072 (28 эпоха) к 60 эпохе  |            0.0185 (84 эпоха)            |
|  Adam   | 0.3064 с 11 эпохи<br/>0.3708 (60 эпоха) | 0.0213 с 33 эпохи<br/>0.0293 (93 эпоха) |                                         |
|  RAdam  | 0.3028 с 20 эпохи<br/>0.3695 (65 эпоха) |                                         | 0.0205 с 51 эпохи<br/>0.0220 (70 эпоха) |

Лучшие F1_macro на тесте на датасете Fruits (остановка при достижении 0.95):

|         | InceptionV3<br/>PyTorch pretrained | InceptionV3<br/>PyTorch not pretrained  |  My InceptionV3   |
|:-------:|:----------------------------------:|:---------------------------------------:|:-----------------:|
| RMSProp |         0.9345 (14 эпоха)          |            0.8703 (30 эпоха)            | 0.7973 (17 эпоха) |
|  Adam   |          0.9636 (2 эпоха)          | 0.9407 с 6 эпохи<br/>0.9505 (12 эпоха)  | 0.9266 (6 эпоха)  |
|  RAdam  |                                    | 0.9400 с 10 эпохи<br/>0.9518 (60 эпоха) |                   |

Получается, что: `RMSProp < RAdam < Adam` на данных датасетах
