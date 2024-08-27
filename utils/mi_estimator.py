import numpy as np
import math
import torch.optim as optim
import torch 
import torch.nn as nn

class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

class CLUBMean(nn.Module):  # Set variance of q(y|x) to 1, logvar = 0. Update 11/26/2022
    def __init__(self, x_dim, y_dim, hidden_size=None):
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        
        super(CLUBMean, self).__init__()
   
        if hidden_size is None:
            self.p_mu = nn.Linear(x_dim, y_dim)
        else:
            self.p_mu = nn.Sequential(nn.Linear(x_dim, int(hidden_size)),
                                       nn.ReLU(),
                                       nn.Linear(int(hidden_size), y_dim))


    def get_mu_logvar(self, x_samples):
        # variance is set to 1, which means logvar=0
        mu = self.p_mu(x_samples)
        return mu, 0
    
    def forward(self, x_samples, y_samples):

        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2.
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2.

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.T_func(torch.cat([x_samples,y_shuffle], dim = -1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)
    


class MI_Minimizer():
    def __init__(self):
        self.device = "cuda"
        self.lr = 3e-4
        self.model = CLUBMean(x_dim=256, y_dim=256, hidden_size=256).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, betas=[0.9, 0.99], weight_decay=0)
        # self.scheduler_mi = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=1e-4)

    def train(self, _neg_pairs):
        self.model.train()
        # print(_neg_pairs[0].shape)
        self.map_i, self.map_j  = _neg_pairs[0], _neg_pairs[1]
        self.map_i, self.map_j = self.map_i.to(self.device), self.map_j.to(self.device)
        
        self.optimizer.zero_grad()
        loglikeli = self.model.learning_loss(self.map_i.detach(), self.map_j.detach())
        # print(loglikeli)
        loglikeli.backward()
        self.optimizer.step()
        mi_loss = self.model(self.map_i, self.map_j)
        return mi_loss.item()
    

class MI_Maximizer():
    def __init__(self):
        self.device = "cuda"
        self.lr = 3e-4
        self.model = MINE(x_dim=256, y_dim=256, hidden_size=256).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, betas=[0.9, 0.99], weight_decay=0)
        # self.scheduler_mi = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=1e-4)

    def train(self, pairs):
        self.model.train()
        # print(_neg_pairs[0].shape)
        self.map_i, self.map_j  = pairs[0], pairs[1]
        self.map_i, self.map_j = self.map_i.to(self.device), self.map_j.to(self.device)
        
        self.optimizer.zero_grad()
        loglikeli = self.model.learning_loss(self.map_i.detach(), self.map_j.detach())
        # print(loglikeli)
        loglikeli.backward()
        self.optimizer.step()
        mi_loss = self.model(self.map_i, self.map_j)
        return mi_loss.item()