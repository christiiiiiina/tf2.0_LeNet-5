import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from mnist_read import load_mnist
print(tf.__version__)

labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']

# 导入数据
DataSetPath = './data/fashion/'
x_test, y_test = load_mnist(DataSetPath, 't10k')
x_test = x_test.reshape((-1, 28, 28, 1))

# 导入模型
new_model = keras.models.load_model('LeNet-5_model.h5')
new_pred = new_model.predict(x_test)

# 预测
for i in range(100):
    np.set_printoptions(suppress=True)  # 取消科学计数法显示
    max_val = np.max(new_pred[i])  # 选出概率最高的预测值
    array = new_pred[i].tolist()  # 转化为列表list
    class_id = array.index(max_val)  # 输出包含概率最高的预测值的位置
    print(class_id, labels[class_id])












