##############################
Might be better using jupyter notebook. 
Using kdd99 dataset. 
Using DNN & pytorch 
built by jaewon YOON 
##############################
import pandas as pd
import numpy as np
import sys
import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import preprocessing

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

#####################################################################################

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class kddDataset(Dataset):
    """ kdd dataset."""

    # Initialize your data, download, etc.
    def __init__(self,Dataset):
        #파일 읽어오기
        df = pd.read_csv(Dataset,header=None, names = col_names)
        self.len = df.shape[0]
        #feature 6개랑 label 값 뽑아오기 
        df_select_six_train = df[['duration','protocol_type','src_bytes','dst_bytes','count','srv_count','label']]
        #normal값 만 1로 만들고 나머지 0으로 만들기 
        df_select_six_train.label = df_select_six_train.label.replace({'normal':1})
        list_count = len(df_select_six_train.label.unique())
        list_=df_select_six_train.label.unique()
        for i, x in enumerate(list_):
            if  list_[i]==0:
                pass
            else:
                df_select_six_train.label = df_select_six_train.label.replace({x:0})
        # LabelEncoder하기
        df_select_six_train.protocol_type=LabelEncoder().fit_transform(df_select_six_train.protocol_type)
        # normalizsing 하기 
        df_select_six_train.iloc[:,0:6]=preprocessing.StandardScaler().fit_transform(df_select_six_train.iloc[:,0:6])
        # feature 값 모으기
        torchTensorsX = df_select_six_train.loc[:,'duration':'srv_count']
        torchTensorsY = df_select_six_train.loc[:,'label']
        # label 값 모으기    
        #torchTensorsX는 feature input 
        torchTensorsX = torch.tensor(torchTensorsX[:].values)
        #torchTensorsY는 label값을 모은 것이다. 
        torchTensorsY = torch.tensor(torchTensorsY[:].values).reshape(-1,1)
    def __getitem__(self, index):
        return torchTensorsX[index], torchTensorsY[index]
        
    def __len__(self):
        return self.len
##################################################
train_dataset = kddDataset("KDDTrain+_2.csv")
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=100,
                          shuffle=True,
                          num_workers=2)
##################################################
test_dataset = kddDataset("KDDTest+_2.csv")
test_loader = DataLoader(dataset=test_dataset,batch_size=100,shuffle=True,num_workers=2)

#### Model define
class Model(torch.nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        # 6 * 10 * 6 * 3 * 1  모델 
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(6, 10)
        self.l2 = torch.nn.Linear(10, 6)
        self.l3 = torch.nn.Linear(6, 3)
        self.l4 = torch.nn.Linear(3,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        out3 = self.sigmoid(self.l3(out2))
        y_pred = self.sigmoid(self.l4(out3))
        return y_pred
model = Model()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
###### Model learning #### 
for epoch in range(10000):
    for i, data in enumerate(train_loader, 0):
        # wrap them in Variable
        inputs, labels = data
        inputs, labels = Variable(inputs).float(), Variable(labels).float()
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(inputs)
        # our model
        # Compute and print loss
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.data[0])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
