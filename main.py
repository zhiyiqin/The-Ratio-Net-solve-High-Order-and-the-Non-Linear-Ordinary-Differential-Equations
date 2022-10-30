import tensorflow as tf 
import logging
import numpy as np 
#from pylab import mpl  
import model
import matplotlib.pyplot as plt
import sys#, getopt
from util import give_batch
import config as C
import time    
#import para3 as P
import base_function as BF
import os
import PIL.Image as I
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif']=['Times New Roman']
#plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False    # 用来正常显示负号
class the_net():
    def __init__(self,train_config,stru_config):
        for item,value in train_config.items():
            print(item)
            print(value)
        self.save_path=train_config['CKPT']       #存储操作
        self.learning_rate=train_config["LEARNING_RATE"]
        self.batch_size=train_config["BATCHSIZE"]
        self.max_iter=train_config["MAX_ITER"]
        self.epoch_save=train_config["EPOCH_SAVE"]     #1
        self.step_each_iter=train_config['STEP_EACH_ITER']   #1000
        self.step_show=train_config['STEP_SHOW']        #30
        self.global_steps = tf.Variable(0,trainable=False)    #定义global_steps的初始值】
        self.stru_config=stru_config      #应该是赋值把
        
        self.sess=tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=1, 
            intra_op_parallelism_threads=1,
        ))
        print("openning sess")
        self.build_net()
        print("building net")
        self.build_opt()
        self.saver=tf.train.Saver(max_to_keep=1)
        self.initialize()
        self.D=give_batch([C.a,C.b])
    
    def get_config(self):
        para_dict=C.para_dict
        equations_coe,value,coe_ret=BF.generate_poly(para_dict)
        self.stru_config["coe"]=coe_ret
        self.stru_config["right_power"]=max(xx[0] for xx \
                in para_dict["left_order_value"])+1#para_dict["d"]-1
        self.stru_config["left_power"]=max(xx[0] for xx \
                in para_dict["right_order_value"])+1#para_dict["d"]-1
        #self.stru_config["left_power"]=1
        #self.stru_config["right_power"]=1
        self.stru_config["xa"]=para_dict["left"]
        self.stru_config["xb"]=para_dict["right"]

    
    def build_net(self):
        self.get_config()
        self.y = model.neural_network(self.stru_config)
        print("开始设置要解的微分方程的形")
        print("传递的时候不应该是用return吗，为什么d_valuep[]可以直接用")
        self.loss_function=C.equation(self.y)

        self.chazhi=(self.y.exact_y-self.y.value)
        print("开始对误差进行处理，便于计算")
        self.loss=tf.reduce_mean(tf.pow(self.loss_function,2))
    def build_opt(self):
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, \
                                self.global_steps, C.decay_step, C.decay_rate, staircase=True)

        self.opt=tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
                .minimize(self.loss,global_step=self.global_steps)
        
    def initialize(self):             
        print("初始化我想多看看")
        ckpt=tf.train.latest_checkpoint(self.save_path)     #自动找到最近保存的变量文件
        if ckpt!=None:                                      #不等于
            self.saver.restore(self.sess,ckpt)              #打开ckpt路径
        else:
            self.sess.run(tf.global_variables_initializer()) #
    def plot(self,real,value,d_exact_value1,d_exact_value2,d_value1,d_value2,cha,epoch):
        plt.plot(value)
        plt.plot(real)
        plt.plot(d_exact_value1)
        plt.plot(d_value1)
        plt.plot(d_exact_value2)
        plt.plot(d_value2)
        #plt.legend(["from %s--Numerical result Mean error-%s"%(self.y.the_model,cha) ,"Analytical result","First order divirative of analytical result","First order divirative of numerical result","Second order divirative of analytical result","Second order divirative of numerical result"])
        if os.path.isdir("pic\\") == False:
            os.mkdir("pic\\")
        plt.savefig("pic\\"+self.stru_config["the_model"]+"_epoch_%s"%epoch)
        plt.close()
        plt.show()
    def log_config(self,log_path,logging_name):
        logger = logging.getLogger(logging_name)
        #logger：日志对象，logging模块中最基础的对象
        #是我们进行日志记录时创建的对象，我们可以调
        #用它的方法传入日志模板和信息，来生成一条条日志记录
        #用logging.getLogger(name)方法进行初始化
        logger.setLevel(level=logging.DEBUG)
        #logger对象的常用方法1：setLevel：设置日志等级
        #一旦设置了日志等级，则调用比等级低的日志记录函数则不会输出
        #debug info warning error critical
        handler = logging.FileHandler(log_path, encoding='UTF-8')
        #logging模块自带的三个handler之一。
        #即用来处理日志记录的类，它可以将 Log Record 输出到我们指定
        #的日志位置和存储形式等，如我们可以指定将日志通过 FTP
        #协议记录到远程的服务器上，Handler 就会帮我们完成这些事情
        handler.setLevel=(logging.INFO)
        #我猜只是对记录到txt的信息进行筛选
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -%(message)s')
        #print("yunxingdaozhe")
        handler.setFormatter(formatter)
        #把设置的加进去
        logger.addHandler(handler)
        #把信息给到txt文件
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)
        #这三句相当于在控制台显示，
        #setlevel是为了选择让什么出现在控制台
        #addhandler是为了输出到txt
        return logger
    def baocun_loss_t(self,loss_shuzhi,epoch):
        loss_shuzhi=np.array(loss_shuzhi)
        print(loss_shuzhi.shape)
        if os.path.isdir("loss_txt\\") == False:
            os.mkdir("loss_txt\\")
        img_name=r'loss_txt\%s_%s'%(epoch,C.stru_config['the_model'])
        with open(img_name+".txt","w",encoding="utf-8")as f:
            for z in range(loss_shuzhi.shape[0]):
                x=loss_shuzhi[z]
                f.write("%s\t%s\t\n"%(z,x))

    def get_trainable_variables(self):       #查看图计算节点的值0
        #return tf.get_collection(tf.GraphKeys.VARIABLES)
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)  
    def train(self):
        st=time.time()
        loss_shuzhi=[]
        intx_list=[]
        for epoch in range(self.max_iter):
            print("train epoch %s of total %s epoches"%(epoch,self.max_iter))
            for step in range(self.max_iter):
                intx=self.D.inter(self.batch_size)     #在前面给定范围，最终输出一个以batch_size给定样本数量的矩阵
                loss,_,gs,l_r=self.sess.run([self.loss,self.opt,self.global_steps,self.learning_rate],\
                        feed_dict={self.y.input:intx})
                if (step+1)%1==0:
                    loss_shuzhi.append(loss)
                    intx=intx.ravel()
                    intx_list.append(intx)
                #print(type(loss_shuzhi))
                #print("loss做成的列表在这里")                  
                if (step+1)%self.step_show==0:
                    print("loss %s,in epoch %s, in step %s \n,in global step %s, \
                            learning rate is %s, taks %s seconds"%\
                            (loss,epoch,step,gs,l_r,time.time()-st))
                    st=time.time()                 
            if (epoch+1)%self.epoch_save==0:
                if os.path.isdir("ckpt\\") == False:
                    os.mkdir("ckpt\\")
                self.saver.save(self.sess, self.save_path+"/check.ckpt")
                int_x=[[x/100.0] for x in np.arange(C.a*100,C.b*100)]
                chazhi,real,value,d_exact_value,d_value,l_r\
                        =self.sess.run([self.chazhi,self.y.exact_y,self.y.value,\
                        self.y.exact_d_values,self.y.d_values,self.learning_rate]\
                        ,feed_dict={self.y.input:int_x})
                chazhi = np.nanmean(chazhi)
                cha='%.3e'%chazhi
                a=len(loss_shuzhi)
                print("loss存储的长度")                
                variables=self.sess.run(self.get_trainable_variables())
                num_list=[]
                for mi in variables:
                    mi=np.array(mi)
                    if len(mi.shape) == 1:
                        num = mi.shape[0]
                        num_list.append(num)
                    else :
                        num_1,num_2 = mi.shape
                        num=num_1*num_2
                        num_list.append(num)
                mk=0
                for i in range(len(num_list)):
                    mk+=num_list[i]
                logger = self.log_config('log.txt',"模型名字%s"%C.stru_config["the_model"])
                logger.info([cha,mk])
                self.baocun_loss_t(loss_shuzhi,epoch)
                self.plot(real,value,\
                        d_exact_value[1],\
                        d_exact_value[2],\
                        d_value[1],d_value[2],cha,epoch)
                print("Model saved in path: %s in epoch %s. learning_rate is %s"\
                        %(self.save_path,epoch,l_r))
            
if __name__=="__main__":
    main_net=the_net(C.train_config,C.stru_config)
    main_net.train()
