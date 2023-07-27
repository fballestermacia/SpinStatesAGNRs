import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh
from TB_GNR import TB_HamArmchair_finite, TB_HamArmchair_infinite
from scipy.optimize import curve_fit, fsolve
import warnings

warnings.filterwarnings('ignore')

def coltopoint(array,column):
    #array: lista con archivos.dat o .txt
    #column: indice de la columna a modificar
    x = np.array([float(i.replace(',', '.'))  
        for i in [(line.split()[column]) 
        for line in array]])
    return x

def fk(kx,ky,t):
    return t*(1+np.exp(1j*kx)*2*np.cos(ky))  
    
def ftosolveq(qalphaytheta,ky,t,length):
    qalpha, theta = qalphaytheta
    kx = np.pi
    return [np.real(np.sinh((2*length)*qalpha + theta)), np.real(np.exp(2*theta)-(fk(kx-1j*qalpha,ky,t)/fk(kx+1j*qalpha,ky,t)))]

def talpha(q,M,t):
    return np.abs(t)*np.sinh(q)/np.sinh((2*M+1)*q)

def Ualpha(q,N,M,U):
    BigM = 4*M+1#2*M+1 #4*M+1
    return U*3*(np.sinh(2*BigM*q)/np.sinh(2*q)-4*np.sinh(BigM*q)/np.sinh(q)+3*BigM)/(N+1)/(np.sinh(BigM*q)/np.sinh(q)-BigM)**2



if __name__=='__main__':
    
    width=13
    t = 1
    U = 1.2*np.abs(t)
    
    qalphas = []
    alphas = np.arange((width+1)/2)+1
    kys = np.pi/(width+1)*alphas[-2:-((width-1)//3)//2-2:-1]

    dummylength = 100
    for ky in kys:
        qalphadummy,thetadummy = fsolve(ftosolveq,[0.8,-10], args=(ky,t,dummylength),xtol=1e-14)
        #print(qalphadummy,thetadummy)
        qalphas.append(qalphadummy)
    
    
    lengths =np.arange(40)+1#np.linspace(1,40,10000)#np.arange(30)+1
    
    Ualphas =[]
    talphas = []
    Ut = []
    
    for length in lengths:
        Udummy = []
        tdummy = []
        for q in qalphas:
            Udummy.append(Ualpha(q,width,length,U))
            tdummy.append(talpha(q,length,t))
        Ualphas.append(Udummy)
        talphas.append(tdummy)
        Ut.append([np.sqrt(1-((2*j/i))**2) for i,j in zip(Udummy,tdummy)])
    
    #print(Ualphas[29])
    
    #ALL IS IN UNITS OF t AND OR U
    
    plt.figure()
    plt.subplot(131)
    plt.ylabel("$t_\\alpha$")
    plt.plot(lengths,talphas,'.-')
    plt.xlabel("$M$")
    plt.subplot(132)
    plt.ylabel("$U_\\alpha$")
    plt.plot(lengths,Ualphas,'.-')
    plt.xlabel("$M$")
    plt.subplot(133)
    plt.ylabel("$m_\\alpha$")
    plt.xlabel("$M$")
    plt.plot(lengths,Ut,'.-')
    plt.ylim(0,1.1)
    
    
    with open("dataMagnetizationProjected\width13attemperature0.001withDanielNotesat22_07_2023-11.18.56.dat") as r:
        rlines = r.readlines()[2:]
        x = coltopoint(rlines,0)
        y = coltopoint(rlines,1)

    with open("dataMagnetization\width13andTemperature0.001at07_07_2023-17.35.24.dat") as r:
        rlines = r.readlines()[2:]
        x2 = coltopoint(rlines,0)
        y2 = coltopoint(rlines,1)
    
    plt.figure()
    plt.xlabel("Length (Unit Cells)")
    plt.ylabel("$m_L$")
    plt.title("AFM Edge Magnetization w={}".format(width))
    plt.scatter(x,y,c='b',marker='s', label="Donwfolding")
    plt.scatter(x2,y2,c='k', label="Full orbital")
    plt.scatter(lengths, np.nansum(Ut, axis=1),c='r',marker='2', label="Analytic")
    plt.legend()
    
    plt.figure()
    plt.xlabel("Length (Unit Cells)")
    plt.ylabel("$m_L$")
    plt.title("AFM Edge Magnetization w={}".format(width))
    plt.scatter(x2,y2,c='k', label="Full orbital")
    plt.ylim(-0.1,(width-1)//6+0.1)
    plt.show()
    
    