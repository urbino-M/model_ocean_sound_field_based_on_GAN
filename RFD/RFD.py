import numpy as np
import math
import matplotlib.pyplot as plt
import h5py
#%% dataread and parameters

# index2 = np.nonzero(data_standard)[1]
# data_standard = data_standard.T.conjugate()

# selected_num = 1

#%% RFD(局部聚焦度计算)
class RFD():
    def cal_RFD(self,data_standard,data_other,dist0):#cal_RFD(self,data_standard,data_other,dist0)
        index_all = []
        self.data_standard = np.array(data_standard)
        data_other = np.array(data_other)
        data_other = data_other.T
        self.index2 = np.nonzero(self.data_standard)[1]
        for i in range(self.index2.shape[0]):
            left_boundary = self.index2[i]-dist0
            right_boundary = self.index2[i]+dist0
            if left_boundary < 1:
                left_boundary = 1
            if right_boundary > data_standard.shape[1]:
                right_boundary = data_standard.shape[1]-1
            mid_index = np.array(range(left_boundary,right_boundary+1))
            index_all = np.append(index_all,mid_index)
        index_all = np.unique(index_all)
        index_all = index_all.astype(int)
        # data_other = data_other.view('complex')
        # data_standard = data_standard.view('complex')
        # data_other = data_other.reshape((data_other.shape[1],1))
        fenzi = np.sum(abs(np.square(data_other[0,index_all])))#只跟第一个做误差分析
        fenmu = np.sum(abs(np.square(data_standard[0,index_all])))
        rfd=fenzi/fenmu
        print(rfd)

        #%% rfd 结果超过最大阈值进行间距调整
        deta = abs(rfd-1)
        threshold1,threshold2=1.05,0.95 #设定允许接受的阈值范围
        num_max=5 #间距最大浮动数值
        flag1 = 0
        num1 = 0
        dist0 = 15
        while rfd > threshold1:
            flag1 = 1
            dist = dist0-1
            num1 = num1+1
            if num1>num_max:
                break
            index_all = []
            for i in range(self.index2.shape[0]): #0-29也是三十次迭代
                left_boundary = self.index2[i]-dist
                right_boundary = self.index2[i]+dist
                if left_boundary < 1:
                    left_boundary = 1
                if right_boundary > data_standard.shape[1]:
                    right_boundary = data_standard.shape[1]-1
                mid_index = np.array(range(left_boundary,right_boundary+1))
                index_all = np.append(index_all,mid_index)
            index_all = np.unique(index_all)
            index_all = index_all.astype(int)
            # data_other = data_other.view('complex')
            # data_standard = data_standard.view('complex')
            fenzi = np.sum(abs(np.square(data_other[0,index_all])))
            fenmu = np.sum(abs(np.square(data_standard[0,index_all])))
            rfd_1=fenzi/fenmu
            if rfd_1<threshold1:
                deta1 = abs(rfd_1-1)
                break
            dist0 = dist
        #%%　rfd 结果小于最小阈值进行间距调整
        flag2 = 0
        num2 = 0
        dist0 = 15
        if flag1==0:
            while rfd < threshold2:
                flag2 = 1
                dist = dist0+1
                num2 = num2+1
                if num2>num_max:
                    break
                index_all = []
                for i in range(self.index2.shape[0]):
                    left_boundary = self.index2[i]-dist
                    right_boundary = self.index2[i]+dist
                    if left_boundary < 1:
                        left_boundary = 1
                    if right_boundary > data_standard.shape[1]:
                        right_boundary = data_standard.shape[1]-1
                    mid_index = np.array(range(left_boundary,right_boundary+1))
                    index_all = np.append(index_all,mid_index)

                index_all = np.unique(index_all)
                index_all = index_all.astype(int)
                # data_other = data_other.view('complex')
                # data_standard = data_standard.view('complex')
                fenzi = np.sum(abs(np.square(data_other[0,index_all])))#只跟第一个做误差分析
                fenmu = np.sum(abs(np.square(data_standard[0,index_all])))
                rfd_2=fenzi/fenmu
                # if rfd_2 >threshold2:
                deta2 = abs(rfd_2-1)
                    # break

                dist0 = dist

        if flag1==1:
            if deta>deta1:
                rfd = rfd_1
        elif flag2==1:
            if deta>deta2:
                rfd = rfd_2
        print(rfd)
        return rfd


# # cal RFD
# data_init_standard = h5py.File('./RFD/c_standard.mat',mode='r')#读取mat数据
# data_standard = data_init_standard['c_standard']
# data_init_other = h5py.File('./RFD/c_other.mat',mode='r')
# data_other = data_init_other['c_other']


# a = data_other[:,1]
# a = a.reshape((3000,1))
# rfd = RFD()
# rfd.cal_RFD(data_standard,a,dist0=15)
