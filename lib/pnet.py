import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA


class P_Net(nn.Module):
    """ Input format is [b,c,h,w] , it outputs norms of output feature map if final is true 
        else outputs feature map of same formate as input.
        The feature map transforms as [b,n,d,d]->[b,k,d,d]
        where n(on of input planes),k(no of output planes) and d(dim of each planes)
    """
   
    def __init__(self,in_units,out_units,units_dim,share_weights=True,final=False,**kwargs):
        super(P_Net, self).__init__(**kwargs)
        self.n = in_units
        self.k = out_units
        self.d = units_dim
        self.final=final
        self.share_weights = share_weights
        
        # module parameters
        self.R = nn.parameter.Parameter(data=torch.empty(1,self.n,self.k),requires_grad=True)
        nn.init.xavier_uniform_(self.R,gain=1.414) # initializing the weights.

        if(self.share_weights):  
            """ W is 'nk-shared' i.e it is shared among k classes and n feature maps"""
            self.W = nn.parameter.Parameter(data=torch.empty(1,self.d,self.d),requires_grad=True)
            nn.init.xavier_uniform_(self.W,gain=1.414) # initializing the weights.
        

    def forward(self, x):
        #R_nor = F.softmax(self.R, dim=1)
        #R_nor = constrain(self.R)
        if(self.share_weights):
            W_s = torch.tile(self.W,[1,self.n,1,1])
            x = torch.matmul(W_s,x)
        x = torch.unsqueeze(x,dim=2)
        x = torch.tile(x,dims=[1, 1, self.k, 1, 1])
        x = torch.mul(x,torch.reshape(self.R, shape=[1, self.n, self.k, 1, 1]))
        x = torch.sum(x,dim=1)
        x = F.relu(x)
        if(self.final):
            x = LA.matrix_norm(x+1e-7,ord='fro',dim=(-2,-1))
        return x
