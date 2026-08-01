'''Time (µs):      0    5    10   15   20   25 is the echo time of flight(by what echo arrives to transducers)
Amplitude:      0    0   0.8   0   0.3   0   is the echo amplitude(hwo strong the echo is)

both constitutes the rf sample data                                                              ''' 
from scatter import *
from wave_propagation import *
import jax.numpy as jnp
echo_amplitude = echo
echo_tof = 2*time_delay
print(echo_amplitude.shape)
print(echo_tof.shape)
rf_data = jnp.stack([echo_tof,echo_amplitude],axis=-1)
print(rf_data)
