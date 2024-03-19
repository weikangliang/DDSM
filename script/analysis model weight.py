# coding=utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)

model_name = "myModel1_test_with_user_tiv"
data_name = "Phones"
model_weights_path = "../dnn_best_model/{}/{}/ckpt_noshuff".format(model_name,data_name)

# 创建一个会话
with tf.Session() as sess:
    # 使用tf.train.import_meta_graph加载模型的计算图和权重
    saver = tf.train.import_meta_graph("../dnn_best_model/{}/{}/ckpt_noshuff.meta".format(model_name,data_name))
    # 恢复模型的权重
    saver.restore(sess, model_weights_path)
    # 获取所有的模型变量
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # 打开一个文件来保存变量的名称和值
    with open("../dnn_best_model/{}/{}/model_weights.txt".format(model_name,data_name), "w") as file:
        # 写入每个变量的名称和值到文件中
        for var in all_vars:
            var_name = var.name
            var_value = sess.run(var)
            # 设置NumPy的打印选项以完整显示数组的内容
            file.write('Variable: {}\n'.format(var_name))
            file.write('Value: \n{}\n'.format(var_value))
            file.write('\n')  # 在变量之间插入一个空行

# 打印完成后提示保存完成
print("变量的名称和值已保存到 model_weights.txt 文件中。")
