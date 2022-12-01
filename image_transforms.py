from multiprocessing import Pool
import os
from PIL import Image


def func_train(f_cls):
    folder = f"fruits-360_dataset/fruits-360/Training"
    print(f_cls + "\n", end="")
    new_cls_folder = f"fruits-360_dataset_299/fruits-360/Training/{f_cls}"
    os.makedirs(new_cls_folder, exist_ok=True)
    for f in os.listdir(f"{folder}/{f_cls}"):
        # print(f"---------{f}")
        if not os.path.isfile(f'{new_cls_folder}/{f}'):
            Image.open(f'{folder}/{f_cls}/{f}').resize((299, 299), Image.ANTIALIAS).save(f'{new_cls_folder}/{f}')


def func_test(f_cls):
    folder = f"fruits-360_dataset/fruits-360/Test"
    print(f_cls + "\n", end="")
    new_cls_folder = f"fruits-360_dataset_299/fruits-360/Test/{f_cls}"
    os.makedirs(new_cls_folder, exist_ok=True)
    for f in os.listdir(f"{folder}/{f_cls}"):
        # print(f"---------{f}")
        if not os.path.isfile(f'{new_cls_folder}/{f}'):
            Image.open(f'{folder}/{f_cls}/{f}').resize((299, 299), Image.ANTIALIAS).save(f'{new_cls_folder}/{f}')


def main():
    print('================ Type:', 'Training')
    folder = f"fruits-360_dataset/fruits-360/Training"
    with Pool(8) as p:
        p.map(func_train, os.listdir(folder))

    print('================ Type:', 'Test')
    folder = f"fruits-360_dataset/fruits-360/Test"
    with Pool(8) as p:
        p.map(func_test, os.listdir(folder))


if __name__ == "__main__":
    main()
