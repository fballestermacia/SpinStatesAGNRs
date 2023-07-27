import numpy as np
import matplotlib.pyplot as plt

def band(kx,ky):
    val = 1+4*np.cos(kx*(3**0.5)/2)**2+4*np.cos(kx*(3**0.5)/2)*np.cos(ky*3/2)
    return val**0.5, -val**0.5



kxs = np.linspace(-np.pi,np.pi,5000)
kys = np.linspace(-np.pi,np.pi,5000)

KX, KY = np.meshgrid(kxs,kys)

pband,mband = band(KX,KY)

fig = plt.figure()
ax = plt.axes(projection ='3d')

ax.plot_surface(KX,KY,pband)
ax.plot_surface(KX,KY,mband)

ax.set_xlabel("$k_x$")
ax.set_ylabel("$k_y$")
ax.set_zlabel("$E(k)$")

plt.show()