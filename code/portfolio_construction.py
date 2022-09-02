"""
Created on Sat Mar  2 08:34:05 2019
Written by Tobias Ingebrigtsen

Asset Pricing Theory, Spring 2019
"""
import requests
import sys
import pandas            as pd
import pandas_datareader as web
import numpy             as np
import matplotlib.pyplot as plt
import datetime          as dt
import yfinance          as yf

from bs4            import BeautifulSoup
from scipy.optimize import minimize


### Define functions ###
def objective_gmvp(w,ret,vCov,rf):
    '''
    Objective function for global mean-variance portfolio
    Inputs:
     - w    : vector of weights
     - ret  : vector of returns
     - vCov : covariance matrix 
     - rf   : risk-free rate
    
    Output:
        - -1s_p : The Sharpe ratio times negative one.
    '''
    var = np.matmul(np.matmul(w,vCov),np.transpose(w))
    std  = var**0.5
    s_p  = (np.matmul(w,ret)-rf)/std
    return -1*s_p


def objective_var(w,vCov):
    '''
    Objective function for minimum variance portfolio.
    Inputs:
     - w   : vector of weights
     - ret : vector of returns
     
    Output:
    - var_p : The portfolio variance.     
    '''
    var_p = np.matmul(np.matmul(w,vCov),np.transpose(w))
    return var_p

def constraint(w):
    '''
    Constraint. We want our weights to sum to one.
    Input:
     - w   : vector of weights
    
    Output:
     - The sum of the weight vector minus one.   
    '''
    return sum(w) - 1


def get_ticks():
    # Get all SP500 tickers and industries
    URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    res       = requests.get(URL).text
    soup      = BeautifulSoup(res,'lxml')
    comp_list = []
    ind_list  = []
    for items in soup.find('table', class_='wikitable').find_all('tr')[1::1]:
        row = items.find_all(['th','td'])
        try:
            comp_list.append(row[0].a.text)
            ind_list.append(row[3].text)
        except: continue
    ind_list = [ind.split('\n')[0] for ind in ind_list ]
    return comp_list, ind_list

###############################################################################

tickers      = ['AAPL','MSFT','GOOGL','GLD']
data         = pd.DataFrame()
start        = dt.datetime(2005, 1, 1)
end          = dt.datetime(2018,1,1)
data         = yf.download(tickers, start=start, end=end).Close
data.columns = tickers
ret_daily    = np.log(data / data.shift(1))
ret_mean_an  = ret_daily.mean()*252
vcov         = ret_daily.cov()*252
print('.'*100)
print('\n Summary statistics \n')
print('.'*100)
print(' \n Annualized mean returns: \n')
print(ret_mean_an)
print('\n Covariance matrix of returns: \n')
print(vcov)
print('.'*100)


port_std = []
port_ret = []
simLen   = 10000
print('\nProgress:')
print('.'*100 + '\n')
for ii in range(1,simLen+1):
    weights   = np.random.normal(0,1,len(tickers))
    weights  /= weights.sum()
    if np.any(weights>=3) or np.any(weights<=-3): continue
    port_var  = np.matmul(np.matmul(weights,vcov),np.transpose(weights))
    port_std  = np.append(port_std,port_var**0.5)
    port_ret  = np.append(port_ret,np.matmul(weights,ret_mean_an))
    if (ii % 100==0 and ii != simLen+1):
        b=('Finished with iteration ' + str(ii) + ' of ' + str(len(range(1,simLen+1))))
        sys.stdout.write('\r'+b) 
    if (ii == simLen): sys.stdout.write('\r'+'-'*30+' Done! '+'-'*30+'\n') 

## Identify mean-variance portfolio through simulation ##
risk_free   = web.get_data_fred('TB3MS',start=end,end=end)/100
s_p         = (port_ret-risk_free['TB3MS'][0])/port_std
s_p_m_ind   = np.argmax(s_p)
min_var_ind = np.argmin(port_std)

### Identify GMVP by constrained optimization ###
rf = risk_free.values[0].tolist()
rf = rf[0]

# Initial guesses
x0   = np.ones(len(weights))
x0  /= x0.sum()

# Run optimization routine, find GMVP
b    = (-3,3)
bnds = (b,)*len(weights)
con  = {'type': 'eq', 'fun': constraint} 
gmvp = minimize(objective_gmvp,x0,args=(ret_mean_an,vcov,rf),constraints=con,
                bounds=bnds,options={'disp': True})
var_gmvp = np.matmul(np.matmul(gmvp.x,vcov),np.transpose(gmvp.x))
ret_gmvp = np.matmul(gmvp.x,ret_mean_an)
std_gmvp = var_gmvp**0.5
s_p_gmvp = (ret_gmvp-rf)/std_gmvp
print('.'*100)
print('\nGlobal Mean Variance portfolio:\n')
print('Return = ' + str(ret_gmvp) + '\n')
print('Standard deviation = ' +  str(std_gmvp) + '\n')
print('Sharpe ratio = ' + str(s_p_gmvp) + '\n')
print('.'*100)


# Run optimization routine, find MVP
minvar = minimize(objective_var,x0,args=(vcov),constraints=con,
                  options={'disp': True})
ret_minvar = np.matmul(minvar.x,ret_mean_an)
std_minvar = minvar.fun**0.5
print('.'*100)
print('\nMinimum variance portfolio:\n')
print('Return = ' + str(ret_minvar) + '\n')
print('Standard deviation = ' + str(std_minvar) + '\n')
print('.'*100)

fig, ax = plt.subplots()
ax.scatter(port_std, port_ret, c='lightblue')
ax.scatter(port_std[s_p_m_ind], port_ret[s_p_m_ind], c='red',marker='D')
ax.scatter(port_std[min_var_ind], port_ret[min_var_ind], c='red',marker='*')
ax.scatter(std_gmvp, ret_gmvp, c='orange',marker='D')
ax.scatter(std_minvar, ret_minvar, c='orange',marker='*')
ax.scatter(np.diag(vcov**0.5), ret_mean_an, c='black',marker='*')
ax.plot([0,port_std[s_p_m_ind]], [risk_free['TB3MS'][0],port_ret[s_p_m_ind]],c='r')
ax.plot([port_std[s_p_m_ind],port_std[s_p_m_ind]*3], [port_ret[s_p_m_ind],\
         (port_ret[s_p_m_ind]-risk_free['TB3MS'][0])*3],c='r',linestyle='--')
plt.xlim(left=0)
ii=1
for port, x, y in zip(['GMVP, scipy','GMVP, sim','Min. Var. Port, scipy','Min. Var. Port, sim'],
                      [std_gmvp,port_std[s_p_m_ind],std_minvar,port_std[min_var_ind]],
                      [ret_gmvp,port_ret[s_p_m_ind],ret_minvar, port_ret[min_var_ind]]):
    plt.annotate(
        port, xy=(x, y), xytext=(-40*((-1)**(ii)), 80*((-1)**(ii))),
        textcoords='offset points', ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=' + str((-1)**(ii)*0.5)))
    ii = ii+1
plt.title("The envelope. Number of assets: " + str(len(tickers)))
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$R_p$')
plt.show()
#plt.savefig('plot1.pdf')


### What if we add more assets? ###

sp_tickers,_ = get_ticks() 

## Fetch 20 random tickers ##
magic_indicies = np.random.randint(0,500,20)
ext_tickers    = [sp_tickers[ind] for ind in magic_indicies]
data_new       = yf.download(ext_tickers, start=start, end=end).Close

ret_daily_new    = np.log(data_new / data_new.shift(1))
ret_mean_an_new  = ret_daily_new.mean()*252
vcov_new         = ret_daily_new.cov()*252
print('.'*100)
print('\n Summary statistics \n')
print('.'*100)
print(' \n Annualized mean returns: \n')
print(ret_mean_an_new)
print('\n Covariance matrix of returns: \n')
print(vcov_new)
print('.'*100)



port_std_new = []
port_ret_new = []
simLen   = 50000
print('\nProgress:')
print('.'*100 + '\n')
for ii in range(1,simLen+1):
    weights   = np.random.normal(0,1,len(data_new.columns))
    weights  /= weights.sum()
    if np.any(weights>=3) or np.any(weights<=-3): continue
    port_var_new = np.matmul(np.matmul(weights,vcov_new),np.transpose(weights))
    port_std_new = np.append(port_std_new,port_var_new**0.5)
    port_ret_new = np.append(port_ret_new,np.matmul(weights,ret_mean_an_new))
    if (ii % 100==0 and ii != simLen):
        b=('Finished with iteration ' + str(ii) + ' of ' + str(len(range(1,simLen+1))))
        sys.stdout.write('\r'+b) 
    if (ii == simLen): sys.stdout.write('\r'+'-'*30+' Done! '+'-'*30+'\n') 
    
## Identify mean-variance portfolio through simulation ##
risk_free       = web.get_data_fred('TB3MS',start=end,end=end)/100
s_p_new         = (port_ret_new-risk_free['TB3MS'][0])/port_std_new
s_p_m_ind_new   = np.argmax(s_p_new)
min_var_ind_new = np.argmin(port_std_new)

# Initial guesses
x0   = np.ones(len(data_new.columns))
x0  /= x0.sum()

# Run optimization routine, find GMVP
b        = (-3,3)
bnds     = (b,)*len(data_new.columns)
con      = {'type': 'eq', 'fun': constraint} 
gmvp_new = minimize(objective_gmvp,x0,args=(ret_mean_an_new,vcov_new,rf),constraints=con,
                    bounds=bnds,options={'disp': True})
ret_gmvp_new = np.matmul(gmvp_new.x,ret_mean_an_new)
std_gmvp_new = objective_var(gmvp_new.x,vcov_new)**0.5

# Run optimization routine, find MVP
minvar_new = minimize(objective_var,x0,args=(vcov_new),constraints=con,
                      options={'disp': True})
ret_minvar_new = np.matmul(minvar_new.x,ret_mean_an_new)
std_minvar_new = minvar_new.fun**0.5


fig, ax = plt.subplots()
ax.scatter(port_std_new, port_ret_new, c='lightblue')
ax.scatter(port_std_new[s_p_m_ind_new], port_ret_new[s_p_m_ind_new], c='red',marker='D')
ax.scatter(port_std_new[min_var_ind_new], port_ret_new[min_var_ind_new], c='red',marker='*')
ax.scatter(std_gmvp_new, ret_gmvp_new, c='orange',marker='D')
ax.scatter(std_minvar_new, ret_minvar_new, c='orange',marker='*')
ax.plot([0,port_std_new[s_p_m_ind_new]], [risk_free['TB3MS'][0],port_ret_new[s_p_m_ind_new]],c='r')
ax.scatter(np.diag(vcov_new**0.5), ret_mean_an_new, c='black',marker='*')
ii=1
for port, x, y in zip(['GMVP, scipy','GMVP, sim','Min. Var. Port, scipy','Min. Var. Port, sim'],
                      [std_gmvp_new,port_std_new[s_p_m_ind_new],std_minvar_new,port_std_new[min_var_ind_new]],
                      [ret_gmvp_new,port_ret_new[s_p_m_ind_new],ret_minvar_new, port_ret_new[min_var_ind_new]]):
    plt.annotate(
        port, xy=(x, y), xytext=(-40*((-1)**(ii)), 80*((-1)**(ii))),
        textcoords='offset points', ha='left', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=' + str((-1)**(ii)*0.5)))
    ii = ii+1
ax.plot([port_std_new[s_p_m_ind_new],port_std_new[s_p_m_ind_new]*3],\
        [port_ret_new[s_p_m_ind_new],(port_ret_new[s_p_m_ind_new]-risk_free['TB3MS'][0])*3],c='r',linestyle='--')
ax.plot([port_std[s_p_m_ind],port_std[s_p_m_ind]*3], [port_ret[s_p_m_ind],\
         (port_ret[s_p_m_ind]-risk_free['TB3MS'][0])*3],c='g',linestyle='--')
plt.xlim(left=0,right=2)
plt.title("The envelope. Number of assets: " + str(len(data_new.columns)))
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$R_p$')
plt.show()
#plt.savefig('plot2.pdf')


### Figure with both envelopes, for comparison
fig, ax = plt.subplots()
ax.scatter(port_std, port_ret, c='red')
ax.scatter(port_std_new, port_ret_new, c='lightblue' ,alpha=0.6)
ax.legend(['Number of assets: '+ str(len(tickers)),'Number of assets: '+ str(len(data_new.columns))])
plt.xlim(left=0,right=1)
plt.title('The envelope.')
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$R_p$')
plt.show()
#plt.savefig('comparison.pdf')

