import torch.nn as nn
import torch
from lib.pnet import *



class efficientnet(nn.Module):

    def __init__(self,efficient_net, num_class=4, classifier='Linear'):
        super(efficientnet, self).__init__()
        #self.efficient_net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        self.net = nn.Sequential(
                                    *list(efficient_net.children())[:-2] # output shape (batch, 320, 4, 4) for input shape [batch,3,128,128]
                                )
        
        if(classifier == 'Linear'):
            self.classifier = nn.Sequential(
                                    nn.AdaptiveAvgPool2d(output_size=1),    
                                    nn.Flatten(),
                                    nn.Linear(320, num_class)                          
                                )
        
        if(classifier == 'P_Net'):
            self.classifier = P_Net(320, num_class, 4, final=True)


    def forward(self, x):
        # features generation from 2nd last-final layer
        features = self.net(x)
        return self.classifier(features)
