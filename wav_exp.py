#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 19:11:10 2021

@author: pciuh
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcol

from sklearn.model_selection import train_test_split

import scipy.stats as scs
import numpy as np

def ztest(x,mp,sp,z_crit,alfa):

    n  = len(x)
    mx = np.mean(x)
    sm = sp/np.sqrt(n)
    zs = (mx-mp)/sm
    pv = 1-scs.norm.cdf(zs)

    print('Samples data')
    print(' Number:',n)
    print('   Mean:',mx)

    # Hypothesis
    if zs >  zsc:
        print('Reject Null Hypothesis')
    else:
        print('Fail to Reject Null Hypothesis')

    #if pv < alfa:
        #print('Reject Null Hypothesis')
    #else:
        #print('Fail to Reject Null Hypothesis')
    return(zs,pv)

def calvar(x,p):
    mu = np.sum(x*p)
    return np.sum(p*(x-mu)**2)

def calcov(x,y,pxy):
    xy = []
    for xx in x:
        xy = np.append(xy,xx*y)
    xy = xy.reshape(-1,y.shape[0])

    print(xy.shape,pxy.shape)
    Ex,Ey,Exy = sum(pxy.T@x).round(4),sum(pxy@y).round(4),np.sum(xy*pxy).round(4)
    Cxx,Cyy = calvar(x,np.sum(pxy,axis=1)),calvar(y,np.sum(pxy,axis=0))
    Cxy = Exy-Ex*Ey
    cov = np.array([[Cxx,Cxy],[Cxy,Cyy]])
    mu  = np.array([Ex,Ey])
    return (cov,mu)

#####
#
#   Scatter diagram of yearly occurence of waves in AMETS 
#   (Atlantinc Marine Energy Test Site) in 2010
#
#####

########
#
# [1]   J. Prendergast, M. Li and W. Sheng, "A Study on the Effects of Wave Spectra on Wave Energy Conversions," in IEEE Journal of Oceanic Engineering, vol. 45, no. 1, pp. 271-283, Jan. 2020, doi: 10.1109/JOE.2018.2869636.
#
########

iDir = 'input/'
pDir = 'plots/'

df = pd.read_csv(iDir + 'wav_scat.csv',sep=';',index_col=0)

#df = df.iloc[:,:-4]
#df.to_csv('wav_scat.csv',sep=';')

print(df.head())
df.index   = df.index.values
df.columns = df.columns.values.astype(float)
df = df.reindex(sorted(df.columns), axis=1)

#df.to_csv('wav_scat_.csv',sep=';')
vhs,vte = df.index.values,df.columns.values.astype(float)
dvt,dvh = np.mean(np.diff(vte)),np.mean(np.diff(-vhs))
pht = df.values

phs = df.sum(axis=1)/1e2
pte = df.sum(axis=0)/1e2

k = np.where(phs == max(phs))[0]
mte,mhs = np.sum(vte*pte),np.sum(vhs*phs)

r,c = np.where(df.values == df.max().max())


cov,mu = calcov(vte,vhs,pht.T/1e2)
ste,shs = np.sqrt(np.diag(cov))
#mu[1] = mhs
#cov
print(df.iloc[r,c])

#mu = [8.75,1.75]

te = [min(vte),max(vte)]
hs = [min(vhs),max(vhs)]

#cov = [[ste,cv],[cv,shs]]
#cov[0,1] = 1.5
#cov[1,0] = 0.5
print('      Mean:',mu)
print('Covariance:',cov)



rnd_seed = 60306
#rnd_seed = 240906
#rnd_seed = None
distr = scs.multivariate_normal(cov = cov, mean = mu, seed = rnd_seed)


nData = 13189 #### acc. to [1]

N     = {'L' : int(0.15*nData), 'S' : 31}

#N[nSmpl]
fdat = distr.rvs(size=nData)

hsg,teg = np.meshgrid(vhs,vte)
fpdf = distr.pdf(fdat)

np.random.seed(rnd_seed)

t_noi = np.random.normal(0.0,0.50,nData)
h_noi = np.random.normal(0.0,0.25,nData)

k = np.where(fdat[:,0]<min(te))[0]
fdat[k,0] = min(te)

k = np.where(fdat[:,1]<min(hs))[0]
fdat[k,1] = min(hs)

#### Adding some noise to data
fndat = fdat + np.array([t_noi,h_noi]).T

k = np.where(fndat[:,0]<min(te))[0]
fndat[k,0] = min(te)

k = np.where(fndat[:,1]<min(hs))[0]
fndat[k,1] = min(hs)

period,height = fndat[:,0],fndat[:,1]
#Hypothesis
# cht = Period / sqrt(Height)
# H0 >> Fully developed seas (oceans, open sea areas): cht >= 3.87
# H1 >> Non-developed seas (offshore sea): cht < 3.87

vLbl = (r'$\mu$:',r'$\sigma$:',r'$n_{SAMPLES}$:',r'$\alpha$:',r'$z_{CRiT}$:',r'$z_{SCORE}$:',r'$p-val$:')
pLbl = ('Mean:','St.Dev.:','Num.Samp.:','Sign.Lvl:','Z.Value:','Z.Score:','p.val')
vFmt = ('%6.2f','%6.2f','%6.0f','%6.2f','%6.2f','%6.2f','%6.2f')

#### Samples for hypothesis test
cht = period / np.sqrt(height)

#### Population data
mcht_p = 4.33
scht_p = np.std(cht)
#mcht_p = mte/np.sqrt(mhs)
#scht_p = ste/np.sqrt(shs)

print('Population data:')
print('Mean:',mcht_p) #,np.mean(period)/np.sqrt(np.mean(height)))
print(' Std:',scht_p)

alf = 5e-2
zsc = scs.norm.ppf(1-alf)


#### Large Sample
t_size = N['L']/nData
_,cht_l = train_test_split(cht,test_size=t_size)
print()
zs,pv = ztest(cht_l,mcht_p,scht_p,zsc,alf)
vVal = (mcht_p,scht_p,len(cht_l),alf,zsc,zs,pv)
dfls = pd.Series(dict(zip(pLbl,vVal))).round(4)
print()

#### Small Sample
t_size = N['S']/nData
_,cht_s = train_test_split(cht,test_size=t_size)
print()
zs,pv = ztest(cht_s,mcht_p,scht_p,zsc,alf)
vVal = (mcht_p,scht_p,len(cht_s),alf,zsc,zs,pv)
dfss = pd.Series(dict(zip(pLbl,vVal))).round(4)
print()
#print(dfss)

dfa = pd.DataFrame({'small':dfss,'large':dfls})
print(dfa)

p_title = 'Estimated Data (Atlantinc Marine Energy Test Site)'
LBL = ('Period, [s]','Wave height, [m]')

xMin,xMax = (0,16)
yMin,yMax = (0,16)
dx,dy  = (4,2/3)

p_linewidth = 2
p_alpha = 0.6
p_size  = 40

cMap  = 'PuRd'


cmap = plt.get_cmap(cMap)
num_colors = 10
hex_col = [mcol.to_hex(cmap(i / num_colors)) for i in range(num_colors)]

p_col = hex_col[4]
l_col = hex_col[-4]

fig,ax = plt.subplots(figsize=(4,3))
flip = dict(markerfacecolor='none',markeredgecolor='none')
boxp = dict(linestyle='-', linewidth=3, color=l_col)
medp = dict(linestyle='-', linewidth=1, color=l_col)
ax.boxplot([cht_s,cht_l],labels=['small','large'],
                 flierprops=flip,boxprops=boxp,medianprops=medp,whiskerprops=medp,capprops=medp)
xLim = ax.get_xlim()
acol = 'tab:gray'
ax.fill_between(xLim,[12,12],[mcht_p,mcht_p],color=acol,alpha=.10,label='open sea')
ax.fill_between(xLim,[0,0],[mcht_p,mcht_p],color=acol,alpha=.30,label='offshore')
ax.legend(frameon=False)
ax.set_ylim(0,12)

fig.savefig(pDir + 'boxplo.png',dpi=150)
teg,hsg = np.meshgrid(vte,vhs)

fig,ax = plt.subplots(figsize=(6,6))
ax.set_title(p_title)
cax = ax.imshow(df.replace(0,np.nan).values, aspect='auto', cmap=cMap, origin='upper',
                extent=[min(vte), max(vte), min(vhs), max(vhs)])

#ax.scatter(mu[0],mu[1],c='k',marker='+')
ax.annotate('%.2f'%mu[0],xytext=(mu[0],mu[0]),xy=(mu[0],mu[1]),ha='center',arrowprops=dict(facecolor='black', arrowstyle='-'))
ax.annotate('%.2f'%mu[1],xytext=(mu[1],mu[1]),xy=(mu[0],mu[1]),va='center',arrowprops=dict(facecolor='black', arrowstyle='-'))
ax.set_xlim(xMin,xMax)
ax.set_ylim(yMin,yMax)
ax.set_xlabel(LBL[0])
ax.set_ylabel(LBL[1])
ax.set_aspect(1)

fig.savefig(pDir + 'scatter.png',dpi=300)


fig,ax = plt.subplots(1,figsize=(9,9))

ax.set_title(p_title)
ax.scatter(fndat[:,0],fndat[:,1],ec='white',marker='s',alpha=p_alpha,
           s=p_size,c=fpdf,cmap=cMap,linewidths=None)
#ax.text(xMin+dx,yMax-dy,'Z-Score test',weight='bold')
#for i,l in enumerate(vLbl):
    #LBL = vFmt[i]%(vVal[i])
    #ax.text(xMin+dx,yMax-(i+2)*dy,LBL,va='top',ha='left')
    #ax.text(xMin+dx,yMax-(i+2)*dy,l,va='top',ha='right')

ax.annotate('%.2f'%mu[0],xytext=(mu[0],mu[0]),xy=(mu[0],mu[1]),ha='center',arrowprops=dict(facecolor='black', arrowstyle='-'))
ax.annotate('%.2f'%mu[1],xytext=(mu[1],mu[1]),xy=(mu[0],mu[1]),va='center',arrowprops=dict(facecolor='black', arrowstyle='-'))
ax.set_xlim(xMin,xMax)
ax.set_ylim(yMin,yMax)
ax.set_xlabel('Wave Period [s]')
ax.set_ylabel('Wave Height [m]')
ax.set_aspect(1)

l,b,w,h=ax.get_position().bounds

ax.set_position([.3,.3,.5,.5])
l,b,w,h=ax.get_position().bounds

nd = np.shape(fdat)[1]
pvec = np.array([[l,.12,w,.1],[.12,b,.1,h]])
argMin = (min(te),min(hs))
argMax = (max(te),max(hs))
hor = ['vertical','horizontal']
xd,yd = [vte,vhs],[pte/dvt,phs/dvh]
for i in range(nd):

    axt = fig.add_axes(pvec[i,:])

    y = fndat[:,i]
    xMin,xMax = (argMin[i],argMax[i])
    std = np.std(y)
    mean = np.mean(y)

    x = np.linspace(xMin,xMax,101)
    pdf = scs.norm.pdf(x,mean,std)

    kwargs = dict(histtype='bar',align='mid',color=l_col,rwidth=1, 
                  alpha=.96, density=True, bins=50, orientation=hor[i])

    ns,_,patches = axt.hist(y,**kwargs)

    if hor[i]=='horizontal':
        axt.plot(pdf,x,'-',color=p_col,lw=p_linewidth)
        axt.scatter(yd[i],xd[i],marker='s', c='none', edgecolors=p_col)
        axt.set_ylim(yMin,yMax)
        axt.invert_xaxis()
    else:
        axt.plot(x,pdf,'-',color=p_col,lw=p_linewidth,label='continuous PDF')
        axt.scatter(xd[i],yd[i],c='none',marker='s', edgecolors=p_col,label='scatter')
        axt.set_xlim(yMin,yMax)
        axt.invert_yaxis()
        axt.legend(frameon=False)

fig.savefig(pDir + 'population.png',dpi=300)
#    fig.savefig('test-'+nSmpl+'.svg',dpi=300)
#    fig.savefig('test-'+nSmpl+'.pdf',dpi=300)
