import argparse

from trainer import Trainer

import os
import numpy as np



data_dir='E:\\研究数据\\西储大学\\sdp_fenlei'
N = 200  # total num instances per class
K_mtr = 84  # total num meta_train classes
K_mva = 25  # total num meta_val classes
K_mte = 25  # total num meta_test classes

x_mtr = np.load(os.path.join(data_dir, 'train_fft_109lei1024.npy'))
x_mtr = np.transpose(x_mtr, (0, 1, 2,3))
x_mtr = np.reshape(x_mtr, [84*200, 1024])

y_tr=np.ones([84*200,1])


x_mva = np.load(os.path.join(data_dir, 'test_fft_109lei1024.npy'))
x_mva = np.transpose(x_mva, (0, 1, 2, 3))
x_mva = np.reshape(x_mva, [25*200,1024])

y_te=np.ones([25*200,1])

'''
from scipy.io import loadmat,savemat
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split

"""处理数据部分"""
class Data_read:
    def __init__(self, snr='None'):

        mat = loadmat('D:\\无监督因果推导下的故障诊断方法\\VAE_TF2\\fft\\fft_xichu_tr')
        mat1 = loadmat('D:\\无监督因果推导下的故障诊断方法\\VAE_TF2\\fft\\fft_xichu_te')
        
        
        #mat = loadmat('D:\\论文mix-cnn\\模型\\西储大学轴承故障\\数据\\10类_-10db_dataset1024.mat')
        #mat = loadmat('D:\\国际会议论文\会议论文资料\\PYshenduxuexi\\Fault_Diagnosis_CNN-master\\Datasets\\data7\\None\\dataset1024.mat')
        self.X_train = mat['X_train']
        self.X_test = mat1['X_train'] 
        self.y_train = self.onehot(np.array(mat['y_train'][:,0],dtype=int))
        self.y_test = np.array(mat1['y_train'][:,0],dtype=int)
        self.y_train = np.array(mat['y_train'][:,0],dtype=int)
        self.y_test = self.onehot(np.array(mat1['y_train'][:,0],dtype=int))
        scaler = MinMaxScaler()
        self.X_train_minmax = scaler.fit_transform(self.X_train.T).T
        self.X_test_minmax = scaler.fit_transform(self.X_test.T).T


    def onehot(self,labels):
        one-hot 编码
        n_sample = len(labels)
        n_class = max(labels) + 1
        onehot_labels = np.zeros((n_sample, n_class))
        onehot_labels[np.arange(n_sample), labels] = 1
        return onehot_labels


    #def add_noise(self,snr):
'MIXCNN训练前准备'
data = Data_read(snr='None')

X_train = data.X_train_minmax 
y_train1 = data.y_train
X_test = data.X_test_minmax # 临时代替
y_test = data.y_test

x_mtr, y_tr,x_mva, y_te= train_test_split(X_test, y_test, test_size=0.5)
'''

def main(args,X_train, y_train1,X_test, y_test1,hidden_size=64):
    model_trainer = Trainer(args,X_train, y_train1,X_test, y_test1,hidden_size=64)
    model_trainer.train()
    model_trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300, 
        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=250, 
        help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, 
        help="Learning rate.")
    

    main(parser.parse_args(),x_mtr, y_tr,x_mva, y_te,hidden_size=64)