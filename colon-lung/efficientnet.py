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

data_path = './lung_colon_image_set/colon_image_sets/'

from torchvision import datasets, transforms

transform = transforms.Compose([transforms.Resize((128,128)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

dataset = datasets.ImageFolder(data_path, transform=transform)

train_dataset,test_dataset = torch.utils.data.random_split(dataset,[9000,1000])

print(dataset.classes)
n_classes = len(dataset.classes)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=128,num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=128,num_workers=2)


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

model_trainer = Model_trainer(dataset.classes, device=device)

# training model

history,prediction = model_trainer.fit(train_loader,test_loader,model,loss,optimizer,scheduler,epochs=60)


# results

print(metrics.classification_report(prediction['y_true'],prediction['y_pred'], digits=3))

# saving results

with open('./effi_colon_history_.pickle', 'wb') as handle:
    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./effi_colon_prediction.pickle', 'wb') as handle:
    pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)
