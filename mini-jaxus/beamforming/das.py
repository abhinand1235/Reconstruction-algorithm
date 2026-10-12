import sys
import jax.numpy as jnp
sys.path.append("..")
sys.path.append("../simulation")
sys.path.append("../processing")
from analytic_signal import r_index
from aperature_weighting import ap_weighted
import numpy as np
import matplotlib.pyplot as plt
r_index = np.array(r_index)
ap_weighted = np.array(ap_weighted)
image=[]
for pixel in range(1600):
    total = 0
    for i in range(128):
        x = r_index[pixel,i]
        if x<0 or x>=2200:
            y=0
            total = total + y
        else:
            y = ap_weighted[i,x]
            total = total + y
    image.append(abs(total))
image = np.array(image).reshape(40, 40)
image = image / np.median(image, axis=0, keepdims=True)  
image = image / np.median(image, axis=1, keepdims=True) 
plt.imshow(image, cmap="gray")
plt.colorbar()
plt.show()
plt.imshow(image, cmap="gray")
plt.colorbar()
plt.show()
