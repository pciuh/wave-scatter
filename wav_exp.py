#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 19:11:10 2021

@author: pciuh
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcol

import scipy.stats as scs
import numpy as np

import seaborn as sns

def calcov(x,y,pxy):
    xy = []
    for xx in x:
        xy = np.append(xy,xx*y)
    xy = xy.reshape(-1,y.shape[0])

    Ex,Ey,Exy = sum(pxy.T@x).round(4),sum(pxy@y).round(4),np.sum(xy*pxy).round(4)

    return (Exy-Ex*Ey,Ex,Ey,Exy)

#####
#
#   Scatter diagram of yearly occurence of waves in AMETS 
#   (Atlantinc Marine Energy Test Site) in 2010
#
#####

########
#
#    J. Prendergast, M. Li and W. Sheng, "A Study on the Effects of Wave Spectra on Wave Energy Conversions," in IEEE Journal of Oceanic Engineering, vol. 45, no. 1, pp. 271-283, Jan. 2020, doi: 10.1109/JOE.2018.2869636.
#
########

df = pd.read_csv('wav_scat_.csv',sep=';',index_col=0)

#vec = np.linspace(4.25,2.75,4)
#for x in vec:
    #df[str(x)] = np.zeros_like(df['5.25'])
#
#print(df.shape)

df.index   = df.index.values
df.columns = df.columns.values.astype(float)
df = df.reindex(sorted(df.columns), axis=1)

#df.to_csv('wav_scat_.csv',sep=';')
vhs,vte = df.index.values,df.columns.values.astype(float)
dvt,dvh = np.mean(np.diff(vte)),np.mean(np.diff(-vhs))
pht = df.values

phs = df.sum(axis=1)/1e2
pte = df.sum(axis=0)/1e2

print(sum(pte*vte))
print(sum(phs*vhs))

vcov = calcov(vhs,vte,pht/1e2)

cv = vcov[0]

mu = [vcov[2],vcov[1]-.0]

shs = np.sqrt(np.sum(phs*(vhs-vcov[1])**2)) #/(len(vhs)-1))
ste = np.sqrt(np.sum(pte*(vte-vcov[2])**2)) #/(len(vte)-1))
print('shs:',shs)
print('ste:',ste)
te = [min(vte),max(vte)]
hs = [min(vhs),max(vhs)]

#te  = [1.0,16.0]   ##### Period range
#hs  = [0.5,15.5]   ##### Wave height range

#mu  = np.array([6.24,2.60])    #### Mean period, mean wave height
#mu  = np.array([9.125,2.61])    #### Mean period, mean wave height
#print(mu[0]/np.sqrt(mu[1]))

cov = [[ste,cv],[cv,shs]]

print('      Mean:',mu)
print('Covariance:',cov)

rnd_seed = 12321
distr = scs.multivariate_normal(cov = cov, mean = mu, seed = rnd_seed)

nSmpl = 'L'
N     = {'L' : 12000, 'S' : 153}

nData = N[nSmpl]
fdat = distr.rvs(size=nData)

hsg,teg = np.meshgrid(vhs,vte)
#fpdf = distr.pdf(np.array([[vte],[vhs]]).T)

fpdf = distr.pdf(fdat)

t_noi = np.random.normal(0.0,0.50,nData)
h_noi = np.random.normal(0.0,0.25,nData)

k = np.where(fdat[:,0]<min(te))[0]
fdat[k,0] = min(te)

k = np.where(fdat[:,1]<min(hs))[0]
fdat[k,1] = min(hs)

fndat = fdat + np.array([t_noi,h_noi]).T

k = np.where(fndat[:,0]<min(te))[0]
fndat[k,0] = min(te)

k = np.where(fndat[:,1]<min(hs))[0]
fndat[k,1] = min(hs)

cht = fndat[:,0]/np.sqrt(fndat[:,1])

#Hypothesis
# cht = Period / sqrt(Height)
# H0 >> Fully developed seas (oceans, open sea areas): cht >= 3.87
# H1 >> Non-developed seas (offshore sea): cht < 3.87

ctc = 3.87

alf = 1e-2
zsc = scs.norm.ppf(1-alf)

mcht = np.mean(cht)
scht = np.std(cht)


z = (mcht-ctc)*np.sqrt(nData)/scht

pr = scs.norm.cdf(z)

st,pval = scs.ttest_1samp(cht,ctc,alternative='greater')

print('Z-Score:',st,', p-val:',pval)

vLbl = (r'$\mu$:',r'$\sigma$:',r'$n_{SAMPLES}$:',r'$\alpha$:',r'$z_{CRiT}$:',r'$z_{SCORE}$:',r'$p-val$:')
pLbl = ('Mean:','St.Dev.:','Num.Samp.:','Sign.Lvl:','Z.Value:','Z.Score:','p.val')
vVal = (mcht,scht,nData,alf,zsc,z,pval)
vFmt = ('%6.2f','%6.2f','%6.0f','%6.2f','%6.2f','%6.2f','%6.2f')

for i,l in enumerate(pLbl):
    FMT = '%20s' + vFmt[i]
    print(FMT % (l,vVal[i]))


dc = dict(zip(pLbl,vVal))
dfo = pd.Series(dc)

dfo.to_csv('Test.csv',sep=';')

LBL = ('Period, [s]','Wave height, [m]')

p_title = 'Estimated Data (Atlantinc Marine Energy Test Site)'
LBL = ('Period, [s]','Wave height, [m]')

xMin,xMax = (min(te),max(te))
xMin,xMax = (0,16)
yMin,yMax = (min(hs),max(hs))
yMin,yMax = (0,16)
dx,dy  = (4,2/3)
p_linewidth = 2
p_col= '#0088cc'
p_alpha = 0.7
p_size  = 20

l_col = np.array([4,4,4])/255
p_col = [x/255 for x in [250,140,80]]
l_col = '#1e1e1e'
p_col = '#f0a06e'
cMap  = 'PuRd'


cmap = plt.get_cmap(cMap)

# Define the number of colors you want to extract
num_colors = 10

# Generate colors and convert to hex
hex_col = [mcol.to_hex(cmap(i / num_colors)) for i in range(num_colors)]


p_col = hex_col[4]
l_col = hex_col[-4]
fig,ax = plt.subplots()

ax.imshow(df,cmap=cMap,extent=(min(vte),max(vte),min(vhs),max(vhs)))
ax.set_xticks(np.linspace(3.75,13.75,6))
ax.set_yticks(np.linspace(3.75,13.75,6))
#ax.set_xticklabels([str(x) for x in vte])
#ax.set_yticklabels([str(x) for x in vhs])
ax.set_aspect(1)

plt.show()


fig,ax = plt.subplots(1,figsize=(9,9))

ax.set_title(p_title)
ax.scatter(fndat[:,0],fndat[:,1],marker='o',alpha=p_alpha,
           s=p_size,c=fpdf,cmap=cMap,linewidths=None)
ax.text(xMin+dx,yMax-dy,'Z-Score test',weight='bold')
for i,l in enumerate(vLbl):
    LBL = vFmt[i]%(vVal[i])
    ax.text(xMin+dx,yMax-(i+2)*dy,LBL,va='top',ha='left')
    ax.text(xMin+dx,yMax-(i+2)*dy,l,va='top',ha='right')

ax.set_xlim(xMin,xMax)
ax.set_ylim(yMin,yMax)
ax.set_xlabel('Wave Period [s]')
ax.set_ylabel('Wave Height [m]')
ax.set_aspect(1)
#ax.grid()

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

    y = fdat[:,i]
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
#        ml,sl,bl = axt.stem(xd[i],yd[i],orientation=hor[i])
#        ml.set_markeredgecolor(p_col)
#        ml.set_markerfacecolor('none')
#        sl.set_color(p_col)
#        bl.set_color('none')
        axt.scatter(yd[i],xd[i],c='none', edgecolors=p_col)
        axt.set_ylim(yMin,yMax)
        axt.invert_xaxis()
        #axt.grid(axis='y')
    else:
        axt.plot(x,pdf,'-',color=p_col,lw=p_linewidth,label='continuous PDF')
#        axt.scatter(xd[i],yd[i],c='white',edgecolors=p_col)
        axt.scatter(xd[i],yd[i],c='none', edgecolors=p_col,label='scatter')
        axt.set_xlim(yMin,yMax)
        axt.invert_yaxis()
        #axt.grid(axis='x')
        axt.legend(frameon=False)
    fig.savefig('test-'+nSmpl+'.png',dpi=300)
    fig.savefig('test-'+nSmpl+'.svg',dpi=300)
    fig.savefig('test-'+nSmpl+'.pdf',dpi=300)
