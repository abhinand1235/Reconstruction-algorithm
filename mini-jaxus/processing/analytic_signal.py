# Convert the RF data to an analytic signal
# RF data:
# time:      0   5   10   15 ...
# amplitude:  0   0   0.8  0.2 ...
# Analytic signal:
# time, amplitude + j * Hilbert(amplitude)
# Each transducer element can receive the same echo at a different time
# because the elements are at different positions.

# Element 1:  ///////
# Element 2:      ///////
# Element 3:          ///////
# Hilbert transform creates a 90° phase-shifted version of each RF signal.
# The original RF + its Hilbert transform form the analytic signal:
# x + jH(x)
# The analytic signal is then used by the beamforming process.
# The beamformer applies the required delay to each transducer element
# so that the echoes line up.
# Element 1:  ///////
# Element 2:  ///////
# Element 3:  ///////
# The aligned signals are then added to form the beamformed signal.
#complex/analytic representation makes the delay operation easier to perform
import sys
sys.path[0]=sys.path[0].replace("processing","simulation")
from recieve import *
import jax.numpy as jnp
import scipy
from scipy.signal import hilbert
k=rf_data[:,:,1]
k1=rf_data[:,:,0]
val= scipy.signal.hilbert(k,axis=1)      #amplitude * hilbert_amp*j
print(val)
#analytic singal conversion
ans = jnp.stack([
    k1,val
],axis = -1) #axis = -1 creates the new dimension at end 1600,128,2
print(ans)
