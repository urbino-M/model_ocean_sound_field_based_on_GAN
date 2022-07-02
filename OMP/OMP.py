import numpy as np
import math
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
# h5py.get_config().default_file_mode
#%% parameters and dataread
# f0=715 #中心频率
# fs=48000 #采样率
# T=1 #脉宽
# # toeplitz_column = 4874
# n=fs*T #信号点数
# t=np.linspace(0,T,n)#0到T随机产生n个数
# s=0.7*np.exp(2*1j*math.pi*f0*t)
#%% 定义托普利兹
class OMP():
     def __init__(self):
         self.fs = 48000
        # self.T  = 0.2
     def toeplitz(self,c,r): #生成托普利茨矩阵
        c=np.array(c)
        r=np.array(r)
        #将c与r的全部元素构成列向量
        m,n=c.shape
        y,z=r.shape
        temp1=[]
        temp2=[]
        for i in range(n):
            for temp in c:
                temp1.append(temp[i])
        for i in range(z):
            for temp in r:
                temp2.append(temp[i])

        c=temp1
        r=temp2

        p=len(r)
        m=len(c)

        x=list(r[p-1:0:-1])
        for i in c:
            x.append(i)

        temp3=np.arange(0,m)
        temp4=np.arange(p-1,-1,-1)

        temp3.shape=(m,1)
        temp4.shape=(1,p)

        ij=temp3+temp4
        t=np.array(x)[ij]

        return t

     def perform_omp(self,s,data):#s是发射信号，data是接收信号。
        data = np.array(data)#数据转array
        toeplitz_column = data.shape[1]-s.shape[1]+1
        # plt.plot(t,s)
        # plt.show()
        extend_zero = np.zeros(shape=(1,toeplitz_column-1),dtype=complex)
        # extend_zero = entend_zero.T
        s1=np.append(s,extend_zero)#补齐接收信号的长度
        o=np.zeros(int(toeplitz_column),dtype=complex)
        s1 = s1.reshape((1,s1.shape[0]))
        o = o.reshape((1,o.shape[0]))
        A=self.toeplitz(s1,o)
        M,N = A.shape[0],A.shape[1]
        theta = np.zeros(shape=(N,1),dtype=complex) #来存储恢复的theta(列向量)
        At = np.zeros(shape=(M,100),dtype=complex) #用来迭代过程中存储A被选择的列
        Pos_theta = np.zeros(shape=(1,100),dtype=complex)#用来迭代过程中存储A被选择的列序号
        r_n = data.conjugate()#初始化残差(residual)为y
        receive2 = data.conjugate() #是原始数据的转置
        for ii in range(100):
            product = A.T.conjugate()@r_n
            product_abs = abs(product)#array 取绝对值或者是模
            val = np.max(product_abs)#找到最大值
            pos = np.argmax(product_abs)#找到最大值索引
            At[:,ii]=A[:,pos] #存储 pos这一列到At的第ii列里面
            Pos_theta[0,ii] = pos #把序号存到Pos_theta中
            pinv_At = np.linalg.pinv(At[:,0:ii+1])#计算伪逆
            theta_ls = np.matmul(pinv_At,receive2)#pinv_At@receive2 计算最小二乘解
            r_n =receive2 - At[:,0:ii+1]@theta_ls#更新残差
        Pos_theta = np.array(Pos_theta)
        Pos_theta = Pos_theta.astype(int)#作为索引,所以要转化成整形
        theta[Pos_theta] = theta_ls #恢复出的theta
        results = abs(theta)/max(abs(theta))
        print('OMP finished')
        # plt.plot(results)
        # plt.show()
        # file_name='./gen_data/enhanced_H.mat'#保存生成的增强信道
        # sio.savemat(file_name,{'enhanced_H':results})
        return results

#%%
# sig_src_init = h5py.File('../sig_src2.mat',mode='r')
# sig_src = sig_src_init['sig_src2']
# sig_src = np.array(sig_src)
# sig_src = sig_src.T
# data_init = sio.loadmat('../gen_data/gen_datas.mat')
# data = data_init['data']
# data = data.T
# data_init = h5py.File('../gen_data/gen_datas.mat',mode='r')#读取mat数据
# data = data_init['data']
#%%
# omp = OMP()
#
# omp.perform_omp(s=sig_src,data=data)







