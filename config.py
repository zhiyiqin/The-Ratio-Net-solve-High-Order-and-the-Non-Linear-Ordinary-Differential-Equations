import tensorflow as tf
import os
import glob
import numpy as np
stru_config={'n_inp':1,
        'the_model':"leg",#"mlp","pade","leg","Ra"
        'struc':[[8,'tanh'],[8,'tanh'],[8,'tanh']],#'tanh','relu'
        'order_up':8,
        "order_down":3,
        "order_leg":6,
        "derive_order":4,
        "order_Ra_h":6,
        "order_Ra_d":4,
        'var_name':'real'}
##########################################################
decay_step=2000
decay_rate=0.8





train_config={
        'CKPT':'ckpt',
        "new_train":True,
        "BATCHSIZE":1000,
        "MAX_ITER":2000,
        'STEP_EACH_ITER':1000,
        'STEP_SHOW':30,
        'EPOCH_SAVE':1,
        "LEARNING_RATE":0.0001,
}
##########################################################
e=2.71828183
pi=3.1415926535898

a=0 #做边界
b=pi/2#右边界

A0=0
#A2=0   左边界2阶导数

B0=1

#B2=48
#给出边界条件
para_dict={"left":a,#对位字典
            "right":b,
            "left_order_value":[[0,A0]],
            "right_order_value":[[0,B0]],
            }
######################################################


def exact(x):
    return x**5 + 2             #tf.sin(x)       

def equation(y):
    return y.d_values[4] -120*y.input

if train_config["new_train"]:
    if glob.glob(train_config["CKPT"]+"/*")!=[]:
       os.system("rm %s/*"%train_config["CKPT"])


