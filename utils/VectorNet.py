import torch
import torch.nn as nn 
import torch.nn.functional as F


class MLP(nn.Module): 
    def __init__(self,size3,hidden): 
        super(MLP,self).__init__()
        self.size3 = size3
        self.hidden = hidden
        self.linear = nn.Linear(size3,hidden)
        self.norm = nn.LayerNorm([hidden])

    def forward(self,input_var, input_mask, size1,size2): 
        x = self.linear(input_var)
        y = self.norm(x)
        out = F.relu(y)
        out = out * (input_mask.float()*2-1) # make activations of invalid states negative
        pool = F.max_pool2d(out,(size2,1),stride=1)
        repeat = pool.repeat(1,1,size2,1)
        node_output = torch.cat([out,repeat],axis = -1)
        return node_output


class SubGraph(nn.Module): 
    def __init__(self,size3, hidden): 
        super(SubGraph,self).__init__()
        self.size3  = size3
        self.MLP1 = MLP(size3,hidden)
        self.MLP2 = MLP(2*hidden,hidden)
        self.MLP3 = MLP(2*hidden,hidden)
        
    def forward(self,input_var, input_mask,size1,size2):
        x = self.MLP1(input_var, input_mask,size1,size2)
        y = self.MLP2(x,input_mask,size1,size2)
        z = self.MLP3(y,input_mask,size1,size2)
        output = F.max_pool2d(z,(size2,1),stride=1)
        return output


class GNN(nn.Module): 
    def __init__(self,size1,size2): 
        super(GNN,self).__init__()
        self.key_weights = nn.Linear(size1,size2)
        self.value_weights = nn.Linear(size1,size2)
        self.query_weights = nn.Linear(size1,size2)

    def forward(self,input_var): 
        keys = self.key_weights(input_var)
        values = self.value_weights(input_var)
        query = self.query_weights(input_var)
    
        attention = torch.matmul(query,keys.transpose(2,3))
        attention_softmax = F.softmax(attention,dim=-1)
        weighted_values = torch.matmul(attention_softmax,values)
        output = weighted_values.sum(dim=2)
        return output   
    