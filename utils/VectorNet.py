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

    def forward(self, input_var, input_mask, size1,size2): 
        # input_var of size [batch_size, num_graphs, num_nodes, dim_node]
        x = input_var[input_mask, :]
        x = self.linear(x)
        x = F.leaky_relu(x)
       # x = self.norm(x)
        
        out = torch.zeros(size=input_mask.shape + (self.hidden,), device = input_var.device)
        out[input_mask, :] = x
        
        
        # max pooling (general)
        out_ = out.clone()
        out_[~input_mask, :] = -float('inf')  
        pool, max_indices = torch.max(out_, dim = 2, keepdim = True)
        pool[~torch.any(input_mask, dim = 2), :, :] = 0

        
        repeat = pool.expand(-1, -1, size2, -1)
        node_output = torch.cat([out,repeat],axis = -1)
        return node_output


class SubGraph(nn.Module): 
    def __init__(self,size3, hidden): 
        super(SubGraph,self).__init__()
        self.size3  = size3
        self.hidden = hidden
        self.MLP1 = MLP(size3,hidden)
        self.MLP2 = MLP(2 * hidden, hidden)
        self.MLP3 = MLP(2 * hidden, hidden)
        
    def forward(self,input_var, input_mask, size1,size2):
        x = self.MLP1(input_var, input_mask, size1, size2)
        y = self.MLP2(x, input_mask, size1, size2)
        z = self.MLP3(y, input_mask, size1, size2)
        
        # max pooling (after relu)
#         z[~input_mask, :] = -1
#         output, _ = torch.max(z, dim = 2)
        
        # max pooling (general)
#         z_ = z.clone()
#         z_[~input_mask, :] = -float('inf')
#         output, _ = torch.max(z_, dim = 2)

        
        output = z[:, :, 0, -self.hidden:]
        output[~torch.any(input_mask, dim = 2), :] = 0
#         output = output / (torch.sum(output ** 2, dim = 2, keepdim = True) ** 0.5 + 1e-6)
        
        return output


class GNN(nn.Module): 
    def __init__(self,size): 
        super(GNN,self).__init__()
        self.key_weights = nn.Linear(size,64)
        self.value_weights = nn.Linear(size,64)
        self.query_weights = nn.Linear(size,64)
        self.multihead_attn = nn.MultiheadAttention(embed_dim = 64, num_heads = 1)

    def forward(self,input_var, input_mask): 
        batch_size, num_nodes, dim_in = input_var.shape
        

        key = input_var / (torch.sum(input_var ** 2, dim = -1, keepdim = True) ** 0.5 + 1e-6)
        query = input_var / (torch.sum(input_var ** 2, dim = -1, keepdim = True) ** 0.5 + 1e-6)
        output, attention_softmax = self.multihead_attn(query = query.transpose(0, 1), 
                                                          key = key.transpose(0, 1), 
                                                          value = input_var.transpose(0, 1), 
                                                          key_padding_mask = ~input_mask)
        output = output.transpose(0, 1)
    
#         keys = self.key_weights(input_var)
#         values = self.value_weights(input_var)
#         query = self.query_weights(input_var)
#         attention = torch.matmul(query, keys.transpose(1, 2))
#         attention[~input_mask[:, None, :].expand(-1, num_nodes, -1)] = -float('inf')
#         attention_softmax = F.softmax(attention,dim = -1)
#         attention_softmax = attention_softmax * input_mask[:, None, :].float()
#         attention_softmax = attention_softmax / (torch.sum(attention_softmax, dim = 2, keepdim = True) + torch.finfo(torch.float).eps)
#         weighted_values = torch.matmul(attention_softmax,values)
#         weighted_values[~input_mask, :] = 0
#         output = weighted_values
        
#         print(keys.shape)
#         print(keys)
#         print(query.shape)
#         print(query)
        
#         print(attention_softmax[0, 0, :])
#         print(torch.sum(attention_softmax[0, 0, :]))
#         print(attention_softmax[0, 1, :])
#         print(torch.sum(attention_softmax[0, 1, :]))
        return output   
    
