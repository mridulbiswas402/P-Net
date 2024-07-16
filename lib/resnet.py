import torch
import torch.nn as nn
from lib.pnet import *
from torchvision import transforms

    
class conv_block(nn.Module):

      def __init__(self, in_channels, reduce=False):
          super(conv_block,self).__init__()
          self.reduce=reduce
          if(reduce):
              self.out_channels=in_channels*2
              self.c1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=2,padding=1) 
              self.blur = transforms.GaussianBlur(3)
              self.skp = nn.Conv2d(in_channels,self.out_channels, kernel_size=1, stride=2,padding=0)
          else:
              self.out_channels = in_channels
              self.c1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1,padding='same')
            
          self.c2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1,padding='same')
          self.bn1 = nn.BatchNorm2d(self.out_channels)
          self.bn2 = nn.BatchNorm2d(self.out_channels)
          self.relu = nn.ReLU(inplace=True)
          
          
      def forward(self, input_tensor):
          x = self.relu(self.bn1(self.c1(input_tensor)))
          x = self.bn2(self.c2(x))
          if(self.reduce):
                x += self.skp(self.blur(input_tensor))
          else:
                x += input_tensor
          x = self.relu(x)
          return x
 

class resnet(nn.Module):
    def __init__(self,archi,input_dim=32,n_classes=10):
        super(resnet,self).__init__()
        
        self.layers = []
        self.in_plane = 8
        self.in_plane_dim = input_dim
        
        self.layers.append(nn.Conv2d(3, self.in_plane, kernel_size=3, stride=1,padding='same'))
        for reps in archi:
            self.layers.append(self._make_layer(reps,reduce=True))
        
        self.layers.append(P_Net(self.in_plane,n_classes,self.in_plane_dim,share_weights=True,final=True))
        
        self.net = nn.Sequential(*self.layers)
        
        
    def forward(self, x):
        x = self.net(x)
        return x
    
    def compute_shapes(self,input_shape):
        shapes = [input_shape]
        x = torch.randn(input_shape)
        for layer in self.layers:
            x = layer(x)
            shapes.append(x.shape)
        return shapes
    
    def _make_layer(self,reps=2,reduce=True):
        layers = []
        strt = 0
        if(reduce):
            layers.append(conv_block(self.in_plane,reduce=True))
            self.in_plane *=2
            self.in_plane_dim = int(self.in_plane_dim/2)
            strt = 1
        for i in range(strt, reps):
            layers.append(conv_block(self.in_plane,reduce=False))

        return nn.Sequential(*layers)

