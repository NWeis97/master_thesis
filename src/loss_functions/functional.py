import torch
from torch.distributions.normal import Normal
import pdb

# Based on Warburg. F. et al. implementation https://arxiv.org/pdf/2011.12663.pdf
def negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin = 0.0):
    # Calc terms for distribution mean
    muA2 = muA**2
    muP2 = muP**2
    muN2 = muN**2
    #varA2 = varA**2
    varP2 = varP**2
    varN2 = varN**2

    # Gaussian approximative mean
    mu = torch.sum(muP2 + varP - muN2 - varN - 2*muA*(muP - muN), dim=0)

    # Calc terms for distribution variance
    T1 = varP2 + 2*muP2 * varP + 2*(varA + muA2)*(varP + muP2) - 2*muA2 * muP2 - 4*muA*muP*varP 
    T2 = varN2 + 2*muN2 * varN + 2*(varA + muA2)*(varN + muN2) - 2*muA2 * muN2 - 4*muA*muN*varN
    T3 = 4*muP*muN*varA
    #T1 = varP*(varP+2*muP2)+varN*(varN+2*muN2)-4*varA*muP*muN
    #T2 = 2*muA*(muA*(muP2+muN2)-2*muP*varP-2*muN*varN)
    #T3 = 2*(varA+muA2)*((varP+muP2)+(varN+muN2))

    # Gaussian approximative variance
    sigma2 = torch.sum(2*T1 + 2*T2 - 2*T3, dim=0)
    #sigma2 = torch.sum(2*(T1-T2+T3), dim=0)
    sigma = sigma2**0.5

    # Check why sigma = nan
    if sum(sigma.isnan())!=0:
        pdb.set_trace()

    # Get probs of postive being closer to anchor than negative by margin 'margin'
    # Changed from +margin to -margin (consider margin a positive number)
    probs = Normal(loc = mu, scale = sigma + 1e-8).cdf(-margin) 
    nll = -torch.log(probs + 1e-8)
    return nll.mean()


def kl_div_gauss(mu_q, var_q, mu_p, var_p):
    # Get dims and squeeze to 1d
    N, D = mu_q.shape
    var_q = var_q.squeeze()
    var_p = var_p.squeeze()
    # kl diverence for isotropic gaussian
    kl = 0.5 * ((var_q / var_p) * D + \
    1.0 / (var_p) * torch.sum(mu_p**2 + mu_q**2 - 2*mu_p*mu_q, axis=0) - D + \
    D*(torch.log(var_p) - torch.log(var_q)))
    return kl.mean()