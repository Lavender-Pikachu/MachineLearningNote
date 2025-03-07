import math
import numpy as np

import pandas as pd
import os
import csv

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from torch.utils.tensorboard import SummaryWriter

def same_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

# 划分数据集
def train_valid_split(data_set, valid_ratio, seed):
    valid_data_size = int(len(data_set) * valid_ratio)
    train_data_size = len(data_set) - valid_data_size
    generator = torch.Generator().manual_seed(seed)
    # train_data, valid_data = random_split(data_set,
    #                                     [train_data_size, valid_data_size],
    #                                     generator=torch.Generator.manual_seed(seed))
    train_data, valid_data = random_split(data_set,
                                          [train_data_size, valid_data_size],
                                          generator=generator)
    # train_data = torch.stack([data_set[i][0] for i in train_data.indices])
    # valid_data = torch.stack([data_set[i][0] for i in valid_data.indices])
    train_data = data_set.iloc[train_data.indices].reset_index(drop=True)
    valid_data = data_set.iloc[valid_data.indices].reset_index(drop=True)
    print(f'train_data size: {len(train_data)} | valid_data size: {len(valid_data)}')
    return train_data, valid_data

# 选择特征
def select_feature(train_data, valid_data, test_data, select_all=True):
    #选择最后一列数据，即真实数据，需要和预测值做比较
    y_train = train_data.iloc[:, -1].values
    y_valid = valid_data.iloc[:, -1].values

    raw_x_train = train_data.iloc[:, :-1].values
    raw_x_valid = valid_data.iloc[:, :-1].values
    raw_x_test = test_data.iloc[:, :-1].values

    # max_feat_idx = raw_x_train.shape[1]
    if select_all:
        # 选择所有特征，去掉第一列id
        feat_idx = list(range(1, raw_x_train.shape[1]-1))
    else :
        feat_idx = [0, 1, 2, 3]

    if select_all:
        print(f'feat_idx {feat_idx}')
    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid

# 数据集
class COVID19Dataset(Dataset):
    def __init__(self, features, targets=None):
        if targets is None:
            self.targets = None
        else:
            self.targets = torch.FloatTensor(targets)

        self.features = torch.FloatTensor(features)

    def __getitem__(self, idx):
        if self.targets is None:
            return self.features[idx]
        else:
            return self.features[idx], self.targets[idx]

    def __len__(self):
        return len(self.features)

class My_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(My_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x

# 参数设置
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
config = {
    'seed': 5201314,
    'select_all': True,
    'valid_ratio': 0.2,
    'n_epochs': 3000,
    'batch_size': 128,
    'learning_rate': 1e-5,
    'early_stop': 400,
    'save_path': './models/model.ckpt'
}

# 训练过程
def train(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    writer = SummaryWriter()
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    n_epochs = config['n_epochs']

    best_loss = math.inf
    step = 0
    early_stop_count = 0

    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x) # 做出预测，前向传播
            loss = criterion(pred, y) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新模型
            step += 1
            loss_record.append(loss.detach().item())

            train_pbar.set_description(f'Epoch[{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()
        loss_record = []

        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch[{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.6f}, Valid loss: {mean_valid_loss:.6f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])
            print('Saving model with loss: {:.6f}'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\n Model is not improving, so we halt the training session.')
            return


same_seed(config['seed'])
train_data = pd.read_csv('./ml2022spring-hw1/covid.train.csv')
test_data = pd.read_csv('./ml2022spring-hw1/covid.test.csv')
# 划分数据集
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])
print(f"""train_data size: {train_data.shape}, valid_data size: {valid_data.shape}, test_data size: {test_data.shape})""")
# 选择特征
x_train, x_valid, x_test, y_train, y_valid = select_feature(train_data, valid_data, test_data, config['select_all'])
print(f'the number of features: {x_train.shape[1]}')
# 构造数据集
train_dataset = COVID19Dataset(x_train, y_train)
valid_dataset = COVID19Dataset(x_valid, y_valid)
test_dataset = COVID19Dataset(x_test)
# 准备Dataloader
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

# model = My_Model(input_dim=x_train.shape[1], hidden_dim=16, output_dim=8).to(device)
# train(train_loader, valid_loader, model, config, device)

# 预测
def predict(test_loader, model, device):
    model.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds

def save_pred(preds, file):
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

model = My_Model(input_dim=x_train.shape[1], hidden_dim=16, output_dim=8).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device)
save_pred(preds, 'pred.csv')