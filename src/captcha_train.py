from gen_captcha import gen_captcha_text_and_image
from gen_captcha import number
from gen_captcha import alphabet
from gen_captcha import ALPHABET

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras import losses
from keras.models import load_model


text, image = gen_captcha_text_and_image()
print("验证码图像channel:", image.shape)  # (60, 160, 3)
# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = len(text)
print("验证码文本最长字符数", MAX_CAPTCHA)  # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐


# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.zeros(shape=[IMAGE_HEIGHT,IMAGE_WIDTH, 1])
        gray[:, :, 0] = np.mean(image, -1)
        return gray
    else:
        return img


""" 
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。 
np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行 
"""

# 文本转向量
char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set)


def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


""" 
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有 
vec = text2vec("F5Sd") 
text = vec2text(vec) 
print(text)  # F5Sd 
vec = text2vec("SFd5") 
text = vec2text(vec) 
print(text)  # SFd5 
"""


# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        gray = convert2gray(image)
        batch_x[i, :] = gray
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


# 定义CNN
def crack_captcha_cnn():

    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(MAX_CAPTCHA * CHAR_SET_LEN, activation='relu'))

    return model


path = ""

# 训练
model = crack_captcha_cnn()
# loss
optimizer = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
model.compile(loss=losses.mean_squared_error, optimizer=optimizer, metrics=['mae', 'acc'])

'''
predict = tf.reshape(model, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
max_idx_p = tf.argmax(predict, 2)
max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
correct_pred = tf.equal(max_idx_p, max_idx_l)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

'''


def train():
    batch_x, batch_y = get_next_batch(10000)
    model.fit(batch_x, batch_y, batch_size=10, epochs=1000)
    model.save("captcha.h5")

train()

'''
def test():
    model = load_model('my_model.h5')
    classes = model.predict(x_test, batch_size=128)

    for _ in range(10):
            text, image = gen_captcha_text_and_image()

            # f = plt.figure()
            # ax = f.add_subplot(111)
            # ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
            # plt.imshow(image)
            # plt.show()

            image = convert2gray(image)
            image = image.flatten() / 255

            predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
            text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
            predict_text = text_list[0].tolist()
            print("正确: {}  预测: {}".format(text, predict_text))
'''


#test()

