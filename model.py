import tensorflow as tf 
import time
import numpy as np
import Legend#图例
import config as C
import base_function as BF


def get_activate(act_name):
    if act_name=="sigmoid":
        return tf.nn.sigmoid
    elif act_name=="tanh":
        return tf.nn.tanh
    elif act_name=="relu":
        return tf.nn.relu
    else:
        print("激活函数配置错误，请检查config.py文件，激活函数从['tanh','relu','sigmoid']")
        exit()

class neural_network:
    def __init__(self,config):
        self.n_input=config['n_inp']
        self.the_model=config["the_model"]
        self.order_leg=config["order_leg"]
        self.struc=config['struc']
        self.var_name=config['var_name']
        self.coe=config['coe']
        self.right_power=config['right_power']
        self.left_power=config['left_power']
        self.xa=config['xa']
        self.xb=config['xb']
        self.order_up=config['order_up']
        self.order_down=config['order_down']
        self.highest   =config["derive_order"] 
        self.order_Ra_h = config["order_Ra_h"]
        self.order_Ra_d = config["order_Ra_d"]
        self.n_output = 1
        self.weight_initialization =  tf.contrib.layers.xavier_initializer()
        
        self.construct_input()
        if self.the_model=="leg":
            self.build_value_leg()
        elif self.the_model=="pade":
            self.build_value_pade()
        elif self.the_model=="Ra":
            self.build_value_ra()        
        else:
            self.build_value_mlp()
        self.build_derivation()
        self.exact_d_values()
    def construct_input(self):
        print("x的占位符")
        self.input=tf.placeholder(tf.float64,[None, self.n_input])
        print(type(self.input))
        #self.input=np.array(self.input)
        #print("=========++++++++++++++++++=====")
        #exit()
    def build_value_leg(self):
        print("开始计算value，即NET(x)")
        self.value=0
        #self.order_leg=10
        self.legends=Legend.give_legend(self.input,self.order_leg)        
        for i in range(1,self.order_leg+1):
            w=tf.get_variable(self.var_name + 'weight_' + str(i),
                    initializer=tf.constant([0.0001/(i**2)],tf.float64),
                    dtype=tf.float64
                    )
            self.value +=w*self.legends[i]
        self.build_bound()
        
    def build_value_ra(self):
        print("开始计算value，即NET(x)")
        num_h=self.order_Ra_h
        num_d=self.order_Ra_d
        print("计算wi和bi")
        hei_h=[]
        for i in range(1,num_h+1):
            w_hi= tf.get_variable('weight_h'+str(i) , 
                                shape=[1,num_h], 
                                initializer=self.weight_initialization, 
                                dtype=tf.float64)
            hei_h.append(w_hi)
        
        hei_d=[]
        for i in range(1,num_d+1):
            w_di= tf.get_variable('weight_d'+str(i) ,
                                shape=[1,num_d],
                                initializer=self.weight_initialization,
                                dtype=tf.float64)
            hei_d.append(w_di)    
        
        bw_h=[]
        for j in range(1,num_h+1):
            b_hj = tf.get_variable('bias_h' +str(j) , 
                                shape=[num_h], 
                                initializer=self.weight_initialization, 
                                dtype=tf.float64)
            bw_h.append(b_hj)                                
        
        bw_d=[]
        for j in range(1,num_d+1):
            b_dj = tf.get_variable('bias_d' +str(j) ,
                                shape=[num_d],
                                initializer=self.weight_initialization,
                                dtype=tf.float64)
            bw_d.append(b_dj)

        print("计算每一层")
        cm_h=[]
        for i in range(num_h):
            ci=tf.add(tf.matmul(self.input, hei_h[i]), bw_h[i])
            cm_h.append(ci)

        cm_d=[]
        for i in range(num_d):
            ci=tf.add(tf.matmul(self.input, hei_d[i]), bw_d[i])
            cm_d.append(ci)
    
            
        print("计算相乘层数")
        
        d1=0.0
        d2=0.0
        for i in range(num_h):
            print("i等于%s"%i)
            if i==0:
                d1=cm_h[0]
            else:
                d1=tf.multiply(d1,cm_h[i])

        for i in range(num_d):
            if i==0:
                d2=cm_d[0]
            else:
                d2=tf.multiply(d2,cm_d[i])        

        wm_h = tf.get_variable('weight_m_h', 
                            shape=[self.order_Ra_h,1], 
                            initializer=self.weight_initialization, 
                            dtype=tf.float64)      
        bm_h = tf.get_variable('bias_m_h', 
                            shape=[1], 
                            initializer=self.weight_initialization, 
                            dtype=tf.float64)
        
        wm_d = tf.get_variable('weight_m_d',
                            shape=[self.order_Ra_d,1],
                            initializer=self.weight_initialization,
                            dtype=tf.float64)
        bm_d = tf.get_variable('bias_m_d',
                            shape=[1],
                            initializer=self.weight_initialization,
                            dtype=tf.float64)
        print("计算value")
        self.value_h=tf.matmul(d1, wm_h) + bm_h 
        self.value_d=tf.matmul(d2, wm_d) + bm_d
        self.value=tf.div(self.value_h, self.value_d)
        self.build_bound()         
    def build_value_pade(self):
        print("建立网络结构")
        self.value_up=0
        self.value_down=0
        for i in range(1,self.order_up+1):
            w = tf.get_variable(self.var_name + 'weight_up' + str(i), 
                                initializer=tf.constant([0.01/i],tf.float64),
                                dtype=tf.float64)
            tmp=w*tf.pow(self.input,i)
            self.value_up +=tmp
        for i in range(1,self.order_down+1):
            w = tf.get_variable(self.var_name + 'weight_down' + str(i), 
                                initializer=tf.constant([0.01/i],tf.float64),
                                dtype=tf.float64)
            tmp=w*tf.pow(self.input,i)
            self.value_down +=tmp
        self.value=tf.divide(self.value_up,self.value_down)
        self.build_bound()
        
    def build_value_mlp(self):
        print("建立网络结构")
        for i,stru in enumerate(self.struc):
            this_num,this_act=stru
            activate=get_activate(this_act)
            if i == 0:
                w = tf.get_variable(self.var_name + 'weight_' + str(0), 
                                    shape=[self.n_input, this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b = tf.get_variable(self.var_name + 'bias_' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer = activate(tf.add(tf.matmul(self.input, w), b))
               
            else:
                w = tf.get_variable(self.var_name + 'weight_' + str(i), 
                                    shape=[self.struc[i-1][0], this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b = tf.get_variable(self.var_name + 'bias_' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer = activate(tf.add(tf.matmul(self.layer, w), b))
                
        w =  tf.get_variable(self.var_name+'weight_' + str(len(self.struc)), 
                            shape=[self.struc[-1][0], self.n_output], 
                            initializer=self.weight_initialization, 
                            dtype=tf.float64)
        b =tf.get_variable(self.var_name+'bias_' + str(len(self.struc)), 
                            shape=[self.n_output], 
                            initializer=self.weight_initialization, 
                            dtype=tf.float64)
        self.value=tf.matmul(self.layer, w) + b
        self.build_bound()
    def build_bound(self):
        print("计算g(x)，计算试探函数value(x)")
        for ll,c in enumerate(self.coe):
            if ll==0:
                g=tf.constant(c)
                continue
            g +=c*self.input**ll
        
        self.exact_y=C.exact(self.input)#tf.exp(-1*self.input)
        self.value=self.value*(self.input-self.xa)**self.left_power\
                *(self.xb-self.input)**self.right_power+g
        #print(self.right_power)
        #print("mb的值在这里")
        #exit()
    def build_derivation(self):
        print("计算试探函数的导数",self.highest)
        #exit()
        self.d_values={}
        for i in range(1,self.highest+1):
            st=time.time()    
            print("导数%s"%i)
            if i==1:
                self.d_values[i]=tf.gradients(self.value,self.input)[0]
            else:
                self.d_values[i]=tf.gradients(self.d_values[i-1],self.input)[0]
            print("用时%s"%(time.time()-st))
    def exact_d_values(self):
        print("计算真实的导数")
        self.exact_d_values={}
        for i in range(1,self.highest+1):
            print("导数%s"%i)
            if i==1:
                self.exact_d_values[i]=tf.gradients(self.exact_y,self.input)[0]
                #print("     -------------   运行到1    ")
                #print(self.exact_d_values[])
            else:
                #print("=============value===========")
                #print(self.exact_d_values[i])
                #exit()
                self.exact_d_values[i]=tf.gradients(self.exact_d_values[i-1],self.input)[0]              
            
if __name__=="__main__":      
    
   stru_config=C.stru_config
   para_dict=P.para_dict
   _,_,coe_ret=BF.generate_poly(para_dict)   #调用的函数
   stru_config["coe"]=coe_ret
   stru_config["right_power"]=para_dict["d"]-1
   stru_config["left_power"]=para_dict["d"]-1
   stru_config["xa"]=para_dict["left"]
   stru_config["xb"]=para_dict["right"]
   y=neural_network(stru_config)
   
