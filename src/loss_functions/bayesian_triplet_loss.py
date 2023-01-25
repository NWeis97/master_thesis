import torch
import torch.nn as nn
import src.loss_functions.functional as LF
import pdb

# Based on Warburg. F. et al. implementation https://arxiv.org/pdf/2011.12663.pdf

class BayesianTripletLoss(nn.Module):
    """ An object that can calculates the Bayesian Triplet Loss for a set of tuples

    Args:
        margin (float):                     The margin between the anchor/positive and 
                                            anchor/negative pairs.
        varPrior (float):                   The variance of the prior
        kl_scale_factor (float, optional):  The fraction to which the KL divergence contributes to
                                            the loss [DEFAULT: 1e-6].
        var_type (str, optional):           'iso' or 'diag' defining the variance type of the 
                                            embeddings [DEFAULT: 'iso']
    """
   
    def __init__(self, margin: float, varPrior: float, kl_scale_factor: float = 1e-6, 
                 var_type: str = 'iso'): 
        
        super(BayesianTripletLoss, self).__init__()
        
        self.margin = torch.tensor(margin).cuda()
        self.varPrior = torch.tensor(varPrior).cuda()
        self.kl_scale_factor = torch.tensor(kl_scale_factor).cuda()
        self.var_type = var_type
    
    def forward(self, x, label):

        # divide x into anchor, positive, negative based on labels
        D, N = x.shape
        nq = torch.sum(label.data == -1).item() # number of tuples
        S = x.size(1) // nq # number of images per tuple including query: 1+1+n
        A = x[:, label.data == -1].permute(1, 0).repeat(1, S - 2).view((S - 2) * nq, D).permute(1, 0) 
        P = x[:, label.data == 1].permute(1, 0).repeat(1, S - 2).view((S - 2) * nq, D).permute(1, 0)
        N = x[:, label.data == 0]

        # Extract dim based on iso og diagonal gaussian
        if self.var_type == 'iso':
            D = A.shape[0]-1
        else:
            D = int(A.shape[0]/2)
            
        # Extract variance and means
        varA = A[D:, :]
        varP = P[D:, :]
        varN = N[D:, :]

        muA = A[:D, :]
        muP = P[:D, :]
        muN = N[:D, :]

        # calculate nll
        nll = LF.negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin=self.margin)

        # KL(anchor|| prior) + KL(positive|| prior) + KL(negative|| prior)
        muPrior = torch.zeros_like(muA, requires_grad = False)
        varPrior = torch.ones_like(varA, requires_grad = False) * self.varPrior
        kl = (LF.kl_div_gauss_iso(muA, varA, muPrior, varPrior) + \
              LF.kl_div_gauss_iso(muP, varP, muPrior, varPrior) + \
              LF.kl_div_gauss_iso(muN, varN, muPrior, varPrior))
       
        return nll + self.kl_scale_factor * kl, nll, kl


    def __repr__(self):
        return self.__class__._Name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'