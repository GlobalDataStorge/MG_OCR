# coding=utf-8
import multiprocessing

import PIL.Image
import os


# 数据处理, 将图片归一化, 转化为待训练/测试图片文件

def resize(im, size=100):
    # size作为小边进行裁剪
    w, h = im.size
    if w < h:
        h = size * h / w
        w = size
    else:
        w = size * w / h
        h = size
    return im.resize((w, h))


def rotate(im, label):
    if label:
        return im.transpose(5 - label % 4)
    return im


def handle_image(image_path):
    file_name = image_path.split("/")[-1]
    im = PIL.Image.open(image_path)
    im = resize(im, IMAGE_SIZE)
    for i in range(4):
        im = rotate(im, i)
        split = file_name.split(".")
        new_file_name = "%s_%d.%s" % ("".join(split[:-1]), i, split[-1])
        save_file_path = os.path.join(PATH_SAVE, new_file_name)
        im.save(save_file_path)
    pass


PATH_ORIGIN = "/Users/zijiao/Documents/WorkSpace/PyCharm/Dogs&Cats/data/train"
PATH_SAVE = "data/train_"
IMAGE_SIZE = 100
MAX_COUNT = -1

pool = multiprocessing.Pool(10)

if __name__ == '__main__':
    if not os.path.exists(PATH_ORIGIN):
        raise IOError("Dir not found for %s" % PATH_ORIGIN)
    if not os.path.exists(PATH_SAVE):
        os.mkdir(PATH_SAVE)
    files = os.listdir(PATH_ORIGIN)
    count = len(files)
    for i, file_name in enumerate(files):
        if i == MAX_COUNT:
            break
        file_path = os.path.join(PATH_ORIGIN, file_name)

        pool.map_async(handle_image, [file_path])
        # handle_image(file_path)
        print "Progress %.2f%% Handle image %s" % (float(i + 1) * 100 / count, file_name)

    pool.close()
    pool.join()
    print "Handle completed."
