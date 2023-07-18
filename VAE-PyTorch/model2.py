# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 22:51:40 2023

@author: Owner
"""

import torch
import torch.nn as nn

from model_util import reparameterization_trick


class OmniglotEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(OmniglotEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.last_hidden_size = 2* 2 * hidden_size

        self.encoder = nn.Sequential(
            # -1, hidden_size, 14, 14
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            # -1, hidden_size, 7, 7
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            # -1, 2*hidden_size, 4, 4
        )
        self.mu_out = nn.Linear( 256, 64)
        self.sigma_out = nn.Linear( 256, 64)

    def forward(self, x):
        # -1, hidden_size, 2, 2
        h = self.encoder(x)
        # -1, hidden_size*2*2
        #h = h.view(-1, self.last_hidden_size)

        return self.mu_out(h), self.sigma_out(h)


class OmniglotDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(OmniglotDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.decoder = nn.Sequential(
            # -1, hidden_size, 4, 4
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            # -1, hidden_size, 7, 7
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            # -1, hidden_size, 14, 14
            # -1, 1, 28, 28
            nn.Linear(512, 1024),
            nn.Sigmoid()
        )

    def forward(self, h):
        # -1, hidden_size, 2, 2
        # -1, 1, 28, 28
        x = self.decoder(h)
        return x


class VAE(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = OmniglotEncoder(hidden_size=hidden_size)
        self.decoder = OmniglotDecoder(hidden_size=hidden_size)
    
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = reparameterization_trick(mu, log_var)   
        
        return self.decoder(z), mu, log_var

    def generate(self, inp):
        x_pred = self.decoder(inp)

        return x_pred
    
    
    
    
    
    
    
    
    
    
    
    
    
    

