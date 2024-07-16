import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from lib.pnet import *

class conv_block(nn.Module):

      def __init__(self, in_channels=12,out_channels=12):
          super(conv_block,self).__init__()
          self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,padding='same', bias = False)
          self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding='same', bias = False)
          self.relu = nn.ReLU(inplace=True)
          self.bn1 = nn.BatchNorm2d(in_channels)
          self.bn2 = nn.BatchNorm2d(out_channels)

      def forward(self, x):
          x = self.conv1(self.relu(self.bn1(x)))
          x = self.conv3(self.relu(self.bn2(x)))
          return x

          
class Transition(nn.Module):

      def __init__(self, in_channels=12, out_channels =12):
          super(Transition,self).__init__()
          self.in_channels = in_channels
          self.out_channels = out_channels

          self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1,padding='same', bias = False)
          self.bn1 = nn.BatchNorm2d(self.in_channels)

      def forward(self, x):
          x = self.conv1(F.relu(self.bn1(x),inplace = True))
          x = F.avg_pool2d(x, 2, stride=2)
          #self.outplane_dim = x.shape[-1]
          return x


class DenseBlock(nn.ModuleDict):

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        growth_rate: int,
    ) -> None:
        super(DenseBlock,self).__init__()
        for i in range(num_layers):
            layer = conv_block(
                in_channels = num_input_features + i * growth_rate,
                out_channels = growth_rate
            )
            self.add_module("conv_block%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    
    def __init__(
        self,
        growth_rate: int = 12,
        block_config = [6, 12, 24, 16],
        num_init_features: int = 32,
        in_feature_dim: int = 32,
        num_classes: int = 10,
        classifier: str = 'Linear'
        ):

        super(DenseNet,self).__init__()
        
        self.final = classifier
      
        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, num_init_features, kernel_size=3, stride=2, padding=1, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                ]
            )
        )
      
        # Each denseblock
        feature_dim = in_feature_dim//2 #due to stride 2 in conv1.
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(in_channels=num_features, out_channels=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2
                feature_dim = feature_dim // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # classifier layer
        if(classifier == 'Linear'):
            self.classifier = nn.Linear(num_features, num_classes)
        
        if(classifier == 'P_Net'):
            self.classifier = P_Net(num_features, num_classes, feature_dim, final=True)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features,inplace = True)

        if(self.final=='Linear'):
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
        elif(self.final=='P_Net'):
            out = self.classifier(out)
        return out

