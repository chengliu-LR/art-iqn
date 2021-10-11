import torch
import torch.nn as nn
import numpy as np

def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class IQN(nn.Module):
    '''implicit quantile network model'''
    def __init__(self, state_size, action_size, layer_size, n_step, seed, layer_type="ff"):
        super(IQN, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.K = 32
        self.N = 8
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi*i for i in range(self.n_cos)]).view(1,1,self.n_cos).to(self.device) # Starting from 0 as in the paper 
        self.head = nn.Linear(self.input_shape[0], layer_size) # cound be a cnn 
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)
        #weight_init([self.head_1, self.ff_1])


    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).to(self.device).unsqueeze(-1) #(batch_size, n_tau, 1)
        cos = torch.cos(taus*self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos, taus
    

    def forward(self, input, num_tau=8):
        """
        Quantile Calculation depending on the number of tau
        
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]
        
        """
        batch_size = input.shape[0]
        
        x = torch.relu(self.head(input))
        cos, taus = self.calc_cos(batch_size, num_tau) # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size*num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.layer_size) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*num_tau, self.layer_size)
        
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        return out.view(batch_size, num_tau, self.action_size), taus
    

    def get_action(self, inputs):
        quantiles, _ = self.forward(inputs, self.K)
        actions = quantiles.mean(dim=1)
        return actions