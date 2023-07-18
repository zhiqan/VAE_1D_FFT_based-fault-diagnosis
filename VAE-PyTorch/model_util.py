import torch
from torch.nn import functional as F

def reparameterization_trick(mu, log_var):
    std = torch.exp(0.5 * log_var)
    epsilon = torch.randn_like(std)
    return epsilon * std  + mu

def loss_fn(x, x_pred, mu, log_var):
    # Expectation of log p(x|z), increase likelihood of model
    recon_loss = F.mse_loss(x, x_pred,reduction='sum')
    # D_KL(q(z|x) || p(z)), encourages q(z|x) to be close to prior p(z)
    #var = torch.pow(torch.exp(log_var),2)
    #kl_loss = -0.5 * torch.sum(1+torch.log(var)-torch.pow(mu,2)-var)
    
    
    kl_loss =-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp())
    #kl_loss =  torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1),  dim = 0)

    # Input size / num_train_examples
    lambda_kl_loss = 0.005
    #10/512
    print("RECON LOSS", recon_loss)
    print("KL LOSS", lambda_kl_loss * kl_loss)
    return recon_loss + lambda_kl_loss * kl_loss