import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#surface brightness
SB_data = pd.read_csv('../Data/SB.csv', delimiter = '\t',header = 'infer')
B_band = data["B"]
radial_SB = data["Radial Distance"]

#rotational curve
gas_data = pd.read_csv('../Data/rot_vel.csv', delimiter = '\t',header = 'infer')
stellar_data = pd.read_csv('../Data/stellar_rot.csv', delimiter = '\t',header = 'infer')
radial_gas = gas_data["Radial Distance"]
radial_stellar = stellar_data["Radial Distance"]
H_beta_rot = gas_data["H_beta"]
OIII_rot = gas_data["OIII"]
stellar_rot = stellar_data["stellar_rot"] 




