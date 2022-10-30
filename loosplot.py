import glob
import matplotlib.pyplot as plt    
import numpy as np 
import config as C
def plot(zhi_leg,zhi_mlp,zhi_ra,epoch):
    plt.plot(zhi_leg,color="b")
    plt.plot(zhi_mlp,color="g")
    plt.plot(zhi_ra,color="y")
    plt.legend(["Legendre net","MLP net","Ratio net"])
    plt.savefig("loss_total_pic\\"+"%s"%epoch)
    plt.close()
    print("--------------------------")
    plt.show()

def duqu(img_name):
    suoyin=[]
    zhi=[]
    for line in open(img_name,"r",encoding="utf-8"):
        z,x=line.strip().split("\t")        
        z=int(z)
        x=float(x)
        suoyin.append(z)
        zhi.append(x)
    print("======================================")
    return suoyin,zhi

def yibai(zhi_leg):
    zhi_leg=np.array(zhi_leg)
    lun=zhi_leg.shape[0] // 100 
    print(lun)
    lun_zhi=[]
    for i in range(lun):
        real = zhi_leg[i*100]
        lun_zhi.append(real)
    return lun_zhi
'''
def yibai(zhi_leg):
    zhi_leg=np.array(zhi_leg)
    zhi_leg=zhi_leg[:400]
    return zhi_leg
'''
if __name__ == "__main__":
    epoch=2
    path=r"loss_txt\%s_*.txt"%epoch
    img_lst=glob.glob(path)
    print(img_lst)
    suoyin_leg,zhi_leg=duqu(img_lst[0])
    suoyin_mlp,zhi_mlp=duqu(img_lst[1])
    suoyin_ra,zhi_ra=duqu(img_lst[2])
    
    
    
    zhi_leg=yibai(zhi_leg)
    zhi_mlp=yibai(zhi_mlp)
    zhi_ra=yibai(zhi_ra)
    
    plot(zhi_leg,zhi_mlp,zhi_ra,epoch)
    
    
    
    print(zhi_leg)