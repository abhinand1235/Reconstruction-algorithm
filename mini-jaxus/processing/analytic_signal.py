#analytic_signal.py

# Convert the RF data to an analytic signal
# RF data:
# time:      0   5   10   15 ...
# amplitude:  0   0   0.8  0.2 ...
# Analytic signal:
#  amplitude + j * Hilbert(amplitude)
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
import jax.numpy as jnp
import scipy
from scipy.signal import hilbert
sys.path.append("..")
sys.path.append("../simulation")
sys.path.append("../core")
sys.path.append("../processing")
from core.transmit import t_sample
from simulation.recieve import echo_tof,echo_amplitude
sample_index = echo_tof/t_sample
r_index = jnp.round(sample_index).astype(jnp.int32)
table_initial = jnp.zeros((128,2200))
print(r_index.shape)    
table = table_initial.at[jnp.arange(128), r_index].add(echo_amplitude, mode="drop")
k=table
#analytic singal conversion
val= scipy.signal.hilbert(k,axis=1)      #amplitude * hilbert_amp*j
if __name__ == "__main__":
  print(val)
