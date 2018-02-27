import random
import tensorflow as tf
import numpy as np
import time

from setup_mnist import MNIST, MNISTModel   #数据集和模型
from l2_attack import CarliniL2                #攻击方法

def show(img):
    """
    可以将本函数改造成将数据存储在图片文件中！！！
    Show MNSIT digits in the console.
    """
    remap = "  .*#" + "#" * 100
    img = (img.flatten() + .5) * 3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i * 28:i * 28 + 28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    生成要对抗的图像和其目标标签
    return（对抗原图像，目标标签）
    :param data: 数据集本文使用minist
    :param samples: 采样个数，指生成对抗样本的原图像个数
    :param targeted: 是否生成目标标签
    :param start: 从验证集的什么位置开始生成对抗样本
    :param inception: 分类器是否为inception
    :return:
    """
    """
    seq:
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:#如果分类器是inception，我们应该至少采样10个样本
                seq = random.sample(range(1, 1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                # np.argmax(data.test_labels[start+i])
                # 说明data.test_labels[start+i]每一个单项都是一个概率向量
                if (j == np.argmax(data.test_labels[start + i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start + i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start + i])
            targets.append(data.test_labels[start + i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


if __name__ == "__main__":
    with tf.Session() as sess:
        data, model = MNIST(), MNISTModel("models/mnist", sess)

        attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)

        inputs, targets = generate_data(data, samples=1, targeted=True,start=0, inception=False)

        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()

        print("Took", timeend - timestart, "seconds to run", len(inputs), "samples.")

        #用来显示结果
        for i in range(len(adv)):
            print("Valid:")
            show(inputs[i])
            print("Adversarial:")
            show(adv[i])

            print("Classification:", model.model.predict(adv[i:i + 1]))

            print("Total distortion:", np.sum((adv[i] - inputs[i]) ** 2) ** .5)