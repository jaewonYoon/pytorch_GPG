'''
evaluation added _
built by jaewonYoon
'''
import torch
import numpy as np
import sys
import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import preprocessing
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pandas as pd

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


class kddDataset(Dataset):
    """ kdd dataset."""

    # Initialize your data, download, etc.
    def __init__(self,Dataset):
        #파일 읽어오기
        df = pd.read_csv(Dataset,header=None, names = col_names)
        self.len = df.shape[0]
        #feature 4개랑 label 값 뽑아오기 
        df_select_six_train = df[['flag','protocol_type','service','logged_in','label']]
        #normal값 만 1로 만들고 나머지 0으로 만들기 
        df_select_six_train.label = df_select_six_train.label.replace({'normal':1})
        list_count = len(df_select_six_train.label.unique())
        list_=df_select_six_train.label.unique()
        list_
        for i, x in enumerate(list_):
            if  list_[i]==1:
                pass
            else:
                df_select_six_train.label = df_select_six_train.label.replace({x:0})
        # LabelEncoder하기
        df_select_six_train.protocol_type=LabelEncoder().fit_transform(df_select_six_train.protocol_type)
        df_select_six_train.flag = LabelEncoder().fit_transform(df_select_six_train.flag)
        df_select_six_train.service = LabelEncoder().fit_transform(df_select_six_train.service)
        # normalizsing 하기 
        df_select_six_train.iloc[:,0:4]=preprocessing.StandardScaler().fit_transform(df_select_six_train.iloc[:,0:4])
        x_data = df_select_six_train.loc[:,'flag':'logged_in']
        y_data = df_select_six_train.loc[:,'label']
        # label 값 모으기    

        torchTensorsX = torch.tensor(x_data[:].values)
        torchTensorsY = torch.tensor(y_data[:].values).reshape(-1,1)
        self.x_data = torchTensorsX
        self.y_data = torchTensorsY
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
        
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


class Model(torch.nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        # 6 * 10 * 6 * 3 * 1  모델 
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(4, 10)
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


print('\n===> Training Start')
##### apply GPU #####
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
if torch.cuda.device_count() > 1:
    print('\n===> Training on GPU!')
    model = nn.DataParallel(model)
##### training model ##### 


for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        # wrap them in Variable
        inputs, labels = data
        inputs, labels = Variable(inputs).float(), Variable(labels).float()
        inputs, labels = inputs.to(device), labels.to(device) 
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
num_of_correct = 0 
num_of_total =0
accuracy = 0
recall = 0
precision = 0
r_denominator =0
r_numerator =0
p_denominator = 0
p_numerator = 0
attack_count =0
for i, data in enumerate(test_loader, 0):
        # wrap them in Variable
        inputs, labels = data
        inputs, labels = Variable(inputs).float(), Variable(labels).float()
        inputs, labels = inputs.to(device), labels.to(device) 
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(inputs)
        y_pred = (y_pred >= 0.5).float()
        # our model
        # 맞은 개수 
        num_of_correct += (y_pred==labels).sum().item()
        # 전체 개수 
        num_of_total += (y_pred==labels).size(0)
        accuracy = num_of_correct / num_of_total
        labels_attack = (labels==0) 
        '''
            recall 계산 
            분자: (y_pred==0)==(labels) 공격 중에  라벨 값이랑 같은 것 
            분모: (labels==0).sum() 실제데이터 중 공격인 것 ()
        '''
        #recall 분모는 TP + FN 
        r_denominator += (labels==0).sum().item()
        #True Positive count => recall numerator 
        attack_count += 0 
        if(i%10==0):
            print(i, attack_count)
        for j in range(len(y_pred)):
            if(y_pred[j]==0 and labels[j]== 0):
                attack_count+=1
        r_numerator =attack_count
        recall = r_numerator / r_denominator
        '''
            precision 계산 
            분자: (y_pred==0)==(labels).sum() 모델이 공격이라 판정한 것 중 진짜 공격인것 
            분모: y_pred ==0
        '''
        p_denominator += (y_pred==0).float().sum().item()
        p_numerator = r_numerator  
        precision =  p_numerator /p_denominator 
        if( i %10 ==0):
            print("attack.size()", attack_count)
            print(i,"th: accuracy : for test: ",accuracy )
            print(i,"th: recall : for test: ", recall)
            print(i,"th: precision : for test: ",precision)

print("============>")
print("test completed: \n")
print("accuaracy: ", accuracy)
print("recall: ", recall)
print("precision: ", precision)
