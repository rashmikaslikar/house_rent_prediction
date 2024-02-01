import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        
        torch.manual_seed(12345)
        self.normalizer = nn.LayerNorm(num_features)
        self.linear1 = torch.nn.Linear(num_features, 100)
        self.linear2 = torch.nn.Linear(100, 1)
        
    def forward(self, x):
        x = self.normalizer(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        x = F.relu(x)
        
        return x
    
class Prediction_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, num_features):
        super(Prediction_Model, self).__init__()
        layers = [nn.Linear(num_features, 100)]
        for _ in range(3):
            layers.append(nn.Linear(100, 50))

        layers.append(nn.Linear(50, 1))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)

