import numpy as np
import struct
from tqdm import tqdm
from PIL import Image
import os


def toimg(data_file, label_file, data_file_size, label_file_size, datas_root):
    # data_file = 'DATAD/train-images-idx3-ubyte'
    # data_file = 'DATAD/t10k-images.idx3-ubyte'
    # It's 47040016B, but we should set to 47040000B
    # data_file_size = 47040016  # 47040016
    data_file_size = str(data_file_size - 16) + 'B'

    data_buf = open(data_file, 'rb').read()

    magic, numImages, numRows, numColumns = struct.unpack_from(
        '>IIII', data_buf, 0)
    datas = struct.unpack_from(
        '>' + data_file_size, data_buf, struct.calcsize('>IIII'))
    datas = np.array(datas).astype(np.uint8).reshape(
        numImages, 1, numRows, numColumns)

    # label_file = 'DATAD/train-labels.idx1-ubyte'
    # label_file = 'DATAD/t10k-labels.idx1-ubyte'
    # It's 60008B, but we should set to 60000B
    # label_file_size = 60008  # 10008
    label_file_size = str(label_file_size - 8) + 'B'

    label_buf = open(label_file, 'rb').read()

    magic, numLabels = struct.unpack_from('>II', label_buf, 0)
    labels = struct.unpack_from(
        '>' + label_file_size, label_buf, struct.calcsize('>II'))
    labels = np.array(labels).astype(np.int64)

    # datas_root = 'mnist_train'
    # datas_root = 'mnist_test'
    if not os.path.exists(datas_root):
        os.mkdir(datas_root)


    for i in tqdm(range(10)):
        file_name = datas_root + os.sep + str(i)
        if not os.path.exists(file_name):
            os.mkdir(file_name)

    for ii in tqdm(range(numLabels)):
        img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
        label = labels[ii]
        file_name = datas_root + os.sep + str(label) + os.sep + \
                    datas_root + "_" + str(ii) + '.png'
        img.save(file_name)

if __name__ == '__main__':

    #train_data
    train_data = 'DATAD/train-images-idx3-ubyte'
    train_label = 'DATAD/train-labels-idx1-ubyte'
    train_data_size = 47040016
    train_label_size = 60008
    toimg(train_data, train_label, train_data_size, train_label_size, "mnist_train")

    #test_data
    val_data = 'DATAD/t10k-images-idx3-ubyte'
    val_label = 'DATAD/t10k-labels-idx1-ubyte'
    val_data_size = 7840016
    val_label_size = 10008
    toimg(val_data, val_label, val_data_size, val_label_size, "mnist_test")