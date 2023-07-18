from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from torchvision.datasets import MNIST
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from model2 import VAE
from model_util import loss_fn
from torch.utils.data import Dataset

device = "cuda"

def plot(x_preds, plot_size):
    """Plot resulting images."""
    x_preds = x_preds.detach().cpu()
            
    fig, axs = plt.subplots(plot_size[0], plot_size[1])
    for  i in range(plot_size[0]):
        for j in range(plot_size[1]):
            axs[i][j].plot(x_preds[plot_size[1] * i + j])

    plt.show()
    
    

class MyData(Dataset):
    def __init__(self, pics, labels):
        self.pics = pics
        self.labels = labels

        # print(len(self.pics.files))
        # print(len(self.labels.files))

    def __getitem__(self, index):
        # print(index)
        # print(len(self.pics))
        assert index < len(self.pics)
        return torch.Tensor(np.array(self.pics[index])), self.labels[index]

    def __len__(self):
        return len(self.pics)

    def get_tensors(self):
        return torch.Tensor(np.array(self.pics)), torch.Tensor(self.labels)

        
class Trainer:
    def __init__(self, args,X_train, y_train1,X_test, y_test1,hidden_size=64):
        self.args = args
        self.hidden_size=hidden_size
        self.vae = VAE(hidden_size=hidden_size)
        self.optim = optim.Adam(self.vae.parameters(), lr=self.args.lr)
        self.train_ds = MyData(X_train, y_train1)
        self.test_ds = MyData(X_test, y_test1)
        
        
        

        self.train_dataloader = DataLoader(
            self.train_ds,
            self.args.batch_size,
            num_workers = 8,
            prefetch_factor=16,
            drop_last = True,
            shuffle=True)

        self.test_dataloader = DataLoader(
            self.test_ds,
            self.args.batch_size,
            num_workers = 8,
            prefetch_factor=16,
            drop_last = True,
            shuffle=True)
        
    def save(self):
        model_dict = {
            "model": self.vae.state_dict(),
            "optimizer": self.optim.state_dict()
        }

        torch.save(model_dict, "model.pt")

    def train(self):
        for i in tqdm(range(self.args.epochs)):
            for img, _ in self.train_dataloader:
                # Make a model prediction
                img = img     
                
                x_pred, mu, log_var = self.vae(img)
                # Train the VAE
                loss = loss_fn(img, x_pred, mu, log_var)
                
                self.optim.zero_grad()
                
                loss.backward()
                
                self.optim.step()

            self.save()

    
    def test(self, plot_size = (9, 9)):
        self.vae.eval()
        
        # Reconstruction test
        print("PLOTING TEST RECONSTRUCTION")
        for img, _ in self.test_dataloader:    
            x_preds, _, _ = self.vae(img)
            
            print("x_preds.shape", x_preds.shape)
            plt.figure(1)
            plot(x_preds, (4, 8))
            plt.figure(2)
            plot(img, (4, 8))
            xx=x_preds.detach().numpy()
            np.save('E:\\研究数据\西储大学\\x_test_preds.npy',xx)
            xx1=img.numpy()
            np.save('E:\\研究数据\西储大学\\xs_test.npy',xx1)
            break
        for img, _ in self.train_dataloader:    
            x_preds, _, _ = self.vae(img)
            print("x_preds.shape", x_preds.shape)
            plt.figure(3)
            plot(x_preds, (4, 8))
            plt.figure(4)
            plot(img, (4, 8))
            xxx=x_preds.detach().numpy()
            np.save('E:\\研究数据\西储大学\\x_train_preds.npy',xxx)
            xxx1=img.numpy()
            np.save('E:\\研究数据\西储大学\\xs_train.npy',xxx1)
            break
        
        # Sample numbers
        #print("PLOTTING RANDOM SAMPLES")
        #x_preds = self.vae.generate(torch.randn(plot_size[0] * plot_size[1], 256))
        #plot(x_preds, plot_size)

        
