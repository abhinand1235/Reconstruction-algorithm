from probe import *
speed = 1540 # soft tissue
density = 1000
acoustic_imp = speed * density # how much a medium resist the propogarion of sound waves
wavelength = speed / center_freq
if __name__ == "__main__":
 print(wavelength)
 
