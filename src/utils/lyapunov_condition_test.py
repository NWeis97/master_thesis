import torch
import matplotlib.pyplot as plt
import pdb
import pandas as pd
import numpy as np



D = 100000000
K = 50

indx = (np.arange(1,10001,1).tolist() +
            np.arange(10010,100010,10).tolist() +
            np.arange(100100,1000100,100).tolist() +
            np.arange(1001000,100001000,1000).tolist())
indx = np.array(indx) - 1

s_prior_mu = 10
s_prior_s = 10

data = torch.empty(torch.Size((len(indx),K)))
for k in range(K):
    # sample means and variances
    mu_a = torch.distributions.Normal(0,s_prior_mu).rsample(torch.Size((D,)))
    mu_p = torch.distributions.Normal(0,s_prior_mu).rsample(torch.Size((D,)))
    mu_n = torch.distributions.Normal(0,s_prior_mu).rsample(torch.Size((D,)))
    s_a = torch.distributions.Normal(0,s_prior_s).rsample(torch.Size((D,)))
    s_p = torch.distributions.Normal(0,s_prior_s).rsample(torch.Size((D,)))
    s_n = torch.distributions.Normal(0,s_prior_s).rsample(torch.Size((D,)))
    
    s_pd = (s_a**2+s_p**2)
    s_nd = (s_a**2+s_n**2)
    
    d_pd = s_pd**(-1)*(mu_a-mu_p)**2
    d_nd = s_nd**(-1)*(mu_a-mu_n)**2
    
    var_hd = 2*(s_pd**2)*(1+2*d_pd) + 2*(s_nd**2)*(1+2*d_nd)
    M3_hd  = 8*(s_pd**3)*(1+3*d_pd) - 8*(s_nd**3)*(1+3*d_nd)
    
    var_total = torch.cumsum(var_hd,-1)**(3/2)
    M3_total = torch.cumsum(M3_hd,-1)

    res = var_total**(-1)*M3_total
    data[:,k] = res[indx]
    
    if (k+1) % 2 == 0:
        print(k+1)
        

data3 = pd.DataFrame(data)
data3['x']=indx
fig, ax = plt.subplots(1,1,figsize=(8,6))
data3.plot(kind='line',legend=False,lw=1,ax=ax,x='x',colormap='RdBu')
ax.plot([0,D],[0,0],linewidth=2,linestyle='--',color='black')
ax.set_xlabel(f'Dimension D',weight='bold',fontsize=14)
ax.set_ylabel('Value of Lyapunov Condition expression',weight='bold',fontsize=14)
ax.set_xscale('log')
ax.set_title('Lyapunov Condition simulation',weight='bold',fontsize=17)
ax.set_yscale('symlog', linthresh=1e-6)
fig.savefig('reports/figures/Lyapunov_condition_test2.png')

