\\\\\\\\\\\\\\怎么跑代码
#运行的方式   这个是运行在window10系统上的
  ##要求的库
		tensorflow1.14
		其他直接安就好
  ##运行方式
	python main.py
	每次更改模型或者改模型里面的参数，请删掉ckpt\\文件夹

#用了哪个微分方程
    在代码文件夹里面的：lizi.jpg  ，其中展示了，微分方程，解析解，和边界条件 


\\\\\\\\\\\\\\怎么配置代码
#怎么将微分方程写入代码

	##精确解的配置
		###                           config.py 55行 exact函数
			def exact(x):
				return x**5 + 2
	注意：这里自变量x就用x来表示，cos等函数用  np.cos()
	
	##微分方程的配置                  config.py 58行 equation函数
			def equation(y):
				return y.d_values[4] -120*y.input
	注意：这里自变量x就用y.input来表示，cos等函数用 tf.cos()，导数用 y.d_values[从一阶导开始]，因变量y用y.value表示
	
	##试探函数的配置 (x-a)**ma(x-b)**xb*net(x) +g(x)  （构建在model.py 的215行）
		
		###g(x)的配置涉及到a,b，y(a),y(b)       config.py  38行
		    a=-1 #做边界
			b=1#右边界
			A0=1
			A1=5#左边界1阶导数
			#A2=0   左边界2阶导数
			B0=3
			B1=5
			para_dict={"left":a,             
            "right":b,
            "left_order_value":[[0,A0],[1,A1]],    #几阶就要几个边界条件
            "right_order_value":[[0,B0],[1,B1]],   #可以不对称
            }
			ma和mb是自动配置的，不用管
		总的逻辑是，g(x)需要在微分方程最高阶也具有边界的性质，所以假如是四阶微分方程
		          ，那么g(x)也需要四阶，所以left_order_value和right_order_value需要
				  放四个条件进去，以达到g(x)中x有四次幂指数来求导。
				  
        ###net(x)的选择与配置
			####config.py  5行
			stru_config={'n_inp':1,
			'the_model':"mlp",#"mlp","pade","leg","Ra"   #选择使用什么net(x)模型
			'struc':[[8,'tanh'],[8,'tanh'],[8,'tanh']],#'tanh','relu' #mlp模型的网络参数
			'order_up':8,           #pade模型的网络参数
			"order_down":3,         #pade模型的网络参数
			"order_leg":6,          #leg模型的网络参数
			"derive_order":4,       #微分方程最高阶
			"order_Ra_h":6,         #Ratio模型的网络参数
			"order_Ra_d":4,         #Ratio模型的网络参数
			'var_name':'real'}
			
\\\\\\\\\\\\\\怎么调代码
#调代码的方式
	1：网络参数，原则上，所有的模型都是参数里面的值越大，拟合能力越强
	2：学习率的策略   config.py  16行
        decay_step=2000    #两千轮衰减一次
		decay_rate=0.8     #衰减为原来的0.8倍
	3：baichsize等参数： config.py  32行
			train_config={
			'CKPT':'ckpt',
			"new_train":True,
			"BATCHSIZE":1000，          #每次放进去的数据量，越高越稳定
			"MAX_ITER":2000,            #每轮训练多少步，一共训练多少epoch
			'STEP_EACH_ITER':1000,      
			'STEP_SHOW':30,             #每多少步，把loss等结果在cmd控制台展示一次
			'EPOCH_SAVE':1,             #每多少轮保存一次
			"LEARNING_RATE":0.0001,     #初始学习率
	       }


\\\\\\\\\\\\\\怎么展示代码的结果
#结果的展示
	loss_txt文件夹：存储了每个epoch的loss值
	log.txt：存储了每个模型的参数量，loss值
	pic文件夹 ：存储了试探函数的原函数，一阶导，二阶导与解析解函数的原函数，一阶导，二阶导的拟合效果
			
			
			
			
			
			
			