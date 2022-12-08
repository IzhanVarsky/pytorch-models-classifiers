from multiprocessing import Pool
import os

import numpy as np
from PIL import Image


def transform_images(input):
    f_cls, folder, out_folder, size = input
    print(f_cls + "\n", end="")
    new_cls_folder = f"{out_folder}/{f_cls}"
    os.makedirs(new_cls_folder, exist_ok=True)
    for f in os.listdir(f"{folder}/{f_cls}"):
        # print(f"---------{f}")
        if not os.path.isfile(f'{new_cls_folder}/{f}'):
            Image.open(f'{folder}/{f_cls}/{f}').resize((size, size), Image.ANTIALIAS).save(f'{new_cls_folder}/{f}')


def main():
    # for phase in ['Training', 'Test']:
    for phase in ['train', 'test']:
        print('================ Type:', phase)
        size = 224
        # folder = f"fruits-360_dataset/fruits-360/{phase}"
        # out_folder = f"fruits-360_dataset_{size}/fruits-360/{phase}"
        folder = f"cars_archive_299/{phase}"
        out_folder = f"cars_archive_{size}/{phase}"
        with Pool(8) as p:
            files = os.listdir(folder)
            p.map(transform_images, zip(files,
                                        np.repeat(folder, len(files)),
                                        np.repeat(out_folder, len(files)),
                                        np.repeat(size, len(files)),
                                        ))


if __name__ == "__main__":
    main()
