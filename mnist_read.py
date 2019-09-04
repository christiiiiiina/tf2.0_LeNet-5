'''
data_path:数据集本地保存路径
kind:数据集类别（训练集：kind='train'，测试集：kind='t10k'）
'''


def load_mnist(data_path, kind='train'):
    import os
    import gzip
    import struct
    import numpy as np

    '''Load MNIST data from `path`'''
    labels_path = os.path.join(data_path, '%s-labels-idx1-ubyte.gz' % kind)

    images_path = os.path.join(data_path, '%s-images-idx3-ubyte.gz' % kind)

    print(images_path, labels_path)
    with gzip.open(labels_path, 'rb') as lbpath:
        data_file = lbpath.read()
        _, nums = struct.unpack_from('>II', data_file, 0)  # 取前2个整数，返回一个元组
        labels = np.frombuffer(data_file, dtype=np.uint8, offset=8)
        print("{} labels:{}".format(kind, nums))

    with gzip.open(images_path, 'rb') as imgpath:
        data_file = imgpath.read()
        _, nums, width, height = struct.unpack_from('>IIII', data_file, 0)  # 取前4个整数，返回一个元组
        images = np.frombuffer(data_file,
                               dtype=np.uint8, offset=16).reshape(len(labels), width, height)
        print("{} datas shape:({},{},{})".format(kind, nums, width, height))
    return images, labels


def read_mnsit_test():
    import matplotlib.pyplot as plt

    class_names = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag',
                   'ankle_boots']
    data_path = "./data/fashion/"
    x_train, y_train = load_mnist(data_path, 'train')
    x_test, y_test = load_mnist(data_path, 't10k')

    fig = plt.figure("fasion-mnsit")
    showCOLS = 5
    showROWS = 5
    for figCnt in range(showCOLS * showROWS):
        fig.add_subplot(showCOLS, showROWS, figCnt + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[figCnt])
        plt.xlabel(class_names[y_train[figCnt]])

    plt.show()


if __name__ == '__main__':
    read_mnsit_test()
    train_datas, trian_labels = load_mnist('./data/fashion/', 'train')
    test_datas, test_labels = load_mnist('./data/fashion/', 't10k')

