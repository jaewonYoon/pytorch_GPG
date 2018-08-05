from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import sys
import sklearn
import torch.utils

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
df = pd.read_csv("KDDTrain+_2.csv", header=None, names = col_names)
df_test = pd.read_csv("KDDTest+_2.csv", header=None, names = col_names)
df['protocol_type']    
target=["duration" , "protocol_type", "src_bytes", "dst_bytes", "count", "srv_count","label"]
def feature_export(target,df):
    target_index=list()
    target_column=list()
    for i in range(len(target)):
        target_index.append(col_names.index(target[i]))
    for i in target_index:
       if(i==0):
           full_test_array = np.array(df.iloc[:,i])
       else:
           target_column=np.array(df.iloc[:,i])
           target_add = target_column.T
           full_test_array = np.column_stack((full_test_array,target_add))
    return full_test_array
def feature_binary(array):
    for i in range(len(array[:,1])):
        if(array[i,1]=='tcp'):
            array[i,1] = 0
        elif(array[i,1]=='udp'):
            array[i,1] = 1
        else:
            array[i:1] = 2
        if(array[i,6]=='normal'):
            array[i,6]=0
        else:
            array[i,6]=1
test_array= feature_export(target,df_test)
train_array = feature_export(target,df)
feature_binary(test_array)
feature_binary(train_array)
test_array
for i in range(len(test_array[:,1])):
    if(test_array[i,1] == 'icmp'):
        test_array[i,1] =2
for i in range(len(train_array[:,1])):
    if(train_array[i,1] == 'icmp'):
        train_array[i,1] = 2
test_arr = np.vstack(test_array[:, :]).astype(np.float)
train_arr  = np.vstack(train_array[:, :]).astype(np.float)

## removec area ## 
'''
x = test_arr[np.random.choice(test_arr.shape[0],100,replace=False),:]
y = train_arr[np.random.choice(train_arr.shape[0],100,replace=False),:]
test_label = x[:,6]
test_feature = x[:,:6]
test_label=test_label.reshape(100,1)
train_label =y[:,6]
train_feature=y[:,:6]
train_label=train_label.reshape(100,1)
'''

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H_1, H_2, H_3, H_4, D_out): 
        super(TwoLayerNet,self).__init__()
        self.linear1 = torch.nn.Linear(D_in,H_1)
        self.linear2 = torch.nn.Linear(H_1,H_2)
        self.linear3 = torch.nn.Linear(H_2,H_3)
        self.linear4 = torch.nn.Linear(H_3,H_4)
        self.linear5 = torch.nn.Linear(H_4,D_out)
    def forward(self,x):
        h_relu1 = self.linear1(x).clamp(min=0)
        h_relu2 = self.linear2(h_relu1).clamp(min=0)
        h_relu3 = self.linear3(h_relu2).clamp(min=0)
        h_relu4 = self.linear4(h_relu3).clamp(min=0)
        y_pred = self.linear5(h_relu4)
        return y_pred

N, D_in, H_1, H_2, H_3,H_4, D_out = 100, 6, 12, 6, 3, 2, 1

testloader = torch.utils.data
dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = TwoLayerNet(D_in, H_1, H_2, H_3, H_4,D_out)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(10000):
    x = train_arr[np.random.choice(train_arr.shape[0],100,replace=False),:]
    train_label =x[:,6]
    train_feature=x[:,:6]
    train_label=train_label.reshape(100,1)
    x = torch.from_numpy(train_feature[:,:6]).float()
    y = torch.from_numpy(train_label[:,:6]).float()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
'''
for t in range(10000):
    x = test_arr[np.random.choice(test_arr.shape[0],100,replace=False),:]
    test_label = x[:,6]
    test_feature = x[:,:6]
    test_label=test_label.reshape(100,1)
    x = torch.from_numpy(test_feature[:,:6]).float()
    y = torch.from_numpy(test_label[:,:6]).float()
    y_pred = model(x)
    loss =criterion(y_pred,y)
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
'''
