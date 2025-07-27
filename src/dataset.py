# dataset 을 불러와서 전처리 하기
# dataset 로드 후 data확인 , 결측치 확인, Normalization ,reshape    
#Split training and valdiation set 이걸 torch Dataset형태로 반환 및 변환.
# transforms는 이미지(Tensor or PIL) 전용
import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Subset
# pytorch Dataset 작업시 pd로 파일을 읽고 전처리후 pyroch Dataset 형태로 변환
import logging

def Load_Dataset(train_data_path, test_data_path = None, batch_size=64):
    logger = logging.getLogger(__name__)
    # 1. Load CSV
    train_data = pd.read_csv(train_data_path)
    test_data = None
    test_dataloader = None
    logger.debug(f"train_data : {train_data.shape}")
    if test_data_path is not None:
        test_data = pd.read_csv(test_data_path)
        logger.debug(f"test_data shape : {test_data.shape}")

    # 2. Split X, Y
    Y_train = train_data['label']
    X_train = train_data.drop(['label'], axis=1)

    # 3. EDA
    # sns.countplot(Y_train)
    # print(Y_train.value_counts())
    # print("X_train null check:\n", X_train.isnull().any().describe())
    # print("test_data null check:\n", test_data.isnull().any().describe())

    # 4. Normalize
    X_train = X_train / 255.0
    if test_data is not None:
        test_data = test_data / 255.0
    
    # 5 . Reshape
    X_train = X_train.values.reshape(-1, 1, 28, 28)
    if test_data is not None:
        test_data = test_data.values.reshape(-1,1, 28, 28)
    
    # 6. Convert to tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    if test_data is not None:
        test_data = torch.tensor(test_data, dtype=torch.float32)

    # 7. Create TensorDataset
    train_dataset = TensorDataset(X_train, Y_train) 
    # test_dataset = (test_data)
    
    # 8. Split train and validation set
    train_size = int(0.8*len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset,val_dataset = random_split(train_dataset,[train_size, val_size])
    
    #9. Create DataLoader
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if test_data is not None:
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    train_dataset = TensorDataset(X_train, Y_train) 
    
    return train_dataset,train_dataloader, val_dataloader, test_dataloader
   

def Sanity_load_Dataset(train_dataset,sanity_data_size, batch_size=16):
    subset= Subset(train_dataset,list(range(sanity_data_size)))
    sanity_data_loader = DataLoader(subset,batch_size= batch_size)
    return sanity_data_loader
    