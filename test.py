# coding=utf-8
import PIL.Image
import tensorflow as tf
import numpy as np
import os


path = "/Users/zijiao/Documents/WorkSpace/PyCharm/Dogs&Cats/data/test"
for i, file in enumerate(os.listdir(path)):
    if i == 1:
        break
    if file.endswith(".jpg"):
        file_path = os.path.join(path, file)
        im = PIL.Image.open(file_path)
        # im.show()
        w, h = im.size
        if w < h:
            h = 100 * h / w
            w = 100
        else:
            w = 100 * w / h
            h = 100
        im = im.resize((w, h))
        im = rotate(im, 3)
        im.show()


