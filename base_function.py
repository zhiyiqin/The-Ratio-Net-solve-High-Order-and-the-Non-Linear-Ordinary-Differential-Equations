import numpy as np
import matplotlib.pyplot as plt
#import para3 as P
#import Legend

#通过变条件，自动给出满足变条件的幂次多项式

def factor(n):
    ret=1
    for i in range(1,n+1):
        ret *=i
    return ret

def generate_Legend(para_dict):
    pass

def generate_poly(para_dict):
    print("计算coe")
    max_order=len(para_dict["left_order_value"])+len(para_dict["right_order_value"])
    coe={}
    left=para_dict["left"]
    right=para_dict["right"]
    for i in range(max_order):
        tmp_left=[]
        tmp_right=[]
        for j in range(max_order):
            if j<i:                   #0
                tmp_left.append(0)     
                tmp_right.append(0) 
            elif j==i:                #j! 
                tmp_left.append(factor(j))
                tmp_right.append(factor(j))
            else:                     
                p=j-i                  
                c=factor(j)/factor(j-i)      #j!%(j-i)!
                tmp_left.append(c*left**p)   #c(left^p)
                tmp_right.append(c*right**p) #c(right^p)
            coe["left_%s"%i]=tmp_left        #在增加coe字典的元素
            coe["right_%s"%i]=tmp_right
    equations_coe=[]
    value=[]
    for order,v in para_dict["left_order_value"]:
            equations_coe.append(coe["left_%s"%order])      #在增加equations_coe和value的元素
            value.append(v) 
    for order,v in para_dict["right_order_value"]:      #在增加equations_coe和value的元素
            equations_coe.append(coe["right_%s"%order])
            value.append(v)
    coe_matrix=np.array(equations_coe)
    coe_ret=np.dot(np.linalg.inv(coe_matrix),value)  
       
    print(equations_coe,value,coe_ret)
    return equations_coe,value,coe_ret       #return代表结束，下面的就不再进行了   如果想要在函数外使用函数计算的数值，使用return 数值名

def give_g(coe,x):
    print("计算g(x)，计算试探函数value(x)")
    for ll,c in enumerate(coe):
        if ll==0:
            g=c
            continue
        g +=c*x**ll
    plt.plot(g)
    plt.savefig("pic/example_of_g.png")
    
if __name__=="__main__":
    e=2.71828183 
    a=0
    b=1
    A0=1
    A1=0.5
    A2=0
    B0=1/e
    B1=e
    B2=0
    para_dict={"left":a,#对位字典
               "right":b,
               "left_order_value":[[0,A0],[2,A2]],
               "right_order_value":[[0,B0],[1,B1]],
               "d":2,
              }

    equations_coe,value,coe_ret=generate_poly(para_dict)
    x=np.linspace(a,b,100)
    give_g(coe_ret,x)
    print("==============")
