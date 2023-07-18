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
            nn.Conv2d(1, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            # -1, hidden_size, 7, 7
            nn.Conv2d(hidden_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            # -1, 2*hidden_size, 4, 4
            nn.Conv2d(hidden_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            # -1, 2*hidden_size, 2, 2
            nn.Conv2d(hidden_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.mu_out = nn.Linear( self.last_hidden_size, self.last_hidden_size)
        self.sigma_out = nn.Linear( self.last_hidden_size, self.last_hidden_size)

    def forward(self, x):
        # -1, hidden_size, 2, 2
        h = self.encoder(x)
        # -1, hidden_size*2*2
        #h = h.view(-1, self.last_hidden_size)
        h = torch.flatten(h, start_dim=1)

        return self.mu_out(h), self.sigma_out(h)


class OmniglotDecoder(nn.Module):
    def __init__(self, hidden_size):
        super(OmniglotDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.decoder = nn.Sequential(
            # -1, hidden_size, 4, 4
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            # -1, hidden_size, 7, 7
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            # -1, hidden_size, 14, 14
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            # -1, 1, 28, 28
            nn.ConvTranspose2d(hidden_size, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, h):
        # -1, hidden_size, 2, 2
        h = h.view(-1, self.hidden_size, 2,2)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    

