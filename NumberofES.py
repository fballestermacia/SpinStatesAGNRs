import numpy as np
import matplotlib.pyplot as plt


widths=np.arange(3,100,2)
t= -1
U = 1*np.abs(t)

Neses = []
for width in widths:
    alphas = np.arange((width+1)/2)+1
    kys = np.pi/(width+1)*alphas[-2:-((width-1)//3)//2-2:-1]

    Nes = 0
    for ky in kys:
        if ky >np.pi/3+0.01 and ky < np.pi/2:
            Nes += 1
    Neses.append(Nes)
    
plt.figure()
plt.plot(widths,Neses,'.-',c='k')
plt.xlabel("Width")
plt.ylabel("Number of ES")
plt.show()