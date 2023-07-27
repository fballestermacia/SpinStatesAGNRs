import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def ftosolve(kx, ky, N, M, beta):
    fk = 1+2*np.cos(ky)*np.exp(1j*kx)
    return 2*M*kx+np.angle(fk)-beta*np.pi
    
N = 41
M = 100
betas = np.arange(2*M)+1
kys = np.pi/(N+1)*(np.arange((N+1/2))+1)
kxss = []
for beta in betas:
    kxs = []
    for ky in kys:
        kx = fsolve(ftosolve,1,args=(ky,N,M,beta))
        if ky < np.pi/2 or kx<=np.pi/2:
            if not np.isclose(kx,np.pi): 
                kxs.append(kx)
    kxss.append(np.array(kxs))

kylinessmooth = np.linspace(0,np.pi/2,200)
kxsssmooth = []
for beta in betas:
    kxs = []
    for ky in kylinessmooth:
        kx = fsolve(ftosolve,1,args=(ky,N,M,beta))
        kxs.append(kx)
    kxsssmooth.append(np.array(kxs))

kxsssmooth = np.array(kxsssmooth)
plt.figure()
for b in betas:
    plt.plot(kxsssmooth[b-1],kylinessmooth,'b')
for ky in kys[:-1]:
    plt.plot([0,np.pi],[ky,ky],'r')

plt.plot([0,np.pi/2], [np.pi/2,np.pi/2],'r')

for b in betas:
    
    plt.scatter(kxss[b-1],kys[:(len(kxss[b-1]))],c='k',zorder=10)

plt.xlim(0,np.pi)
plt.ylim(0,np.pi/2+1e-2)
plt.xticks([0,np.pi/4,np.pi/2,np.pi*3/4,np.pi],["$0$","$\\frac{\\pi}{4}$","$\\frac{\\pi}{2}$","$\\frac{3\\pi}{4}$","$\\pi$"])
plt.yticks([0,np.pi/6,np.pi/3,np.pi/2],["$0$","$\\frac{\\pi}{6}$","$\\frac{\\pi}{3}$","$\\frac{\\pi}{2}$"])
plt.xlabel("$k_x$")
plt.ylabel("$k_y$")
plt.show()
