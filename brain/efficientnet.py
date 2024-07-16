import sys
sys.path.append('/home/cmater/Mridul')

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from lib.efficientnet import *
from lib.pnet import *
from lib.trainer import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(device)


# data loading and preprocessing.

train_data_path = './Training'
test_data_path = './Testing'

train_transform = transforms.Compose([transforms.Resize((128,128)),
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(p=0.2),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

test_transform = transforms.Compose([transforms.Resize((128,128)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_dataset = datasets.ImageFolder(train_data_path, transform=train_transform)
test_dataset = datasets.ImageFolder(test_data_path, transform=test_transform)


n_classes = len(train_dataset.classes)
print(train_dataset.classes)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)


# model creation
effnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
model = efficientnet(effnet,num_class=n_classes,classifier='P_Net')

x = torch.randn(1,3,128,128)
output = model(x)
print(f'model output shape: {output.shape}.')

model = model.to(device)

total_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable parameters: {total_params}.')
#summary(model,(3,32,32))


# optimizer and loss setting.

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                  mode='min', 
                                                  factor=0.5, 
                                                  patience=2, 
                                                  threshold=0.001,
                                                  min_lr=1e-7,
                                                  verbose=True)

model_trainer = Model_trainer(train_dataset.classes, device=device)

# training model

history,prediction = model_trainer.fit(train_loader,test_loader,model,loss,optimizer,scheduler,epochs=60)


# results

print(metrics.classification_report(prediction['y_true'],prediction['y_pred'], digits=3))

# saving results

with open('./effi_brain_history_.pickle', 'wb') as handle:
    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./effi_brain_prediction.pickle', 'wb') as handle:
    pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)
