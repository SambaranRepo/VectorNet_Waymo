import torch
from torch import nn

# Define model
class Decoder(nn.Module):
    def __init__(self, n_features, n_modes, T):
        super(Decoder, self).__init__()
        
        self.n_modes = n_modes
        self.T = T
        
        self.linear1 = nn.Linear(n_features, 4096)
        self.linear2 = nn.Linear(512, n_modes * T * 2)
        self.linear3 = nn.Linear(512, n_modes)
        self.linear4 = nn.Linear(4096,512)
        
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        # Regression head
        pos = self.linear2(x)
        
        # Classification head
        prob = self.linear3(x)
        prob = torch.sum(prob, axis=0)
        prob = prob - torch.max(prob)
        prob = self.softmax(prob)
        
        return pos.view((-1, self.n_modes, self.T, 2)), prob
