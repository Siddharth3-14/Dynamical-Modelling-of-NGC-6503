import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt

#surface brightness
SB_data_mod = pd.read_csv('../Data/E_model_SB.csv', delimiter = '\t',header = 'infer')
radial_SB_mod = np.array(SB_data_mod["Radial Distance"])
B_band_mod = np.array(SB_data_mod["B"])



SB_data = pd.read_csv('../Data/SB.csv', delimiter = '\t',header = 'infer')
radial_SB = np.array(SB_data["Radial Distance"])
B_band = np.array(SB_data["B"])


total_SB = pm.Model()
with total_SB:

	#priors
	sigma = pm.HalfNormal("sigma" , sigma = 0.4)
	R_d = pm.Gamma("R_d" , alpha = 3, beta = 1)
	mu_0_disk = pm.Gamma("mu_0_disk" , alpha = 3, beta = 1)



	mu = mu_0_disk + 1.0875*(radial_SB_mod/R_d)



	#likelihood
	Y_obs = pm.Normal('Y_obs', mu = mu, sigma = sigma, observed = B_band_mod)
	step = pm.Metropolis() 
	trace = pm.sample(draws = 50000, step = step, tune = 20000, cores = 2)

print(pm.summary(trace))
parameter_mean = pm.summary(trace)["mean"]
print("central luminosty of disk ",10**((27-parameter_mean[2])/2.5)) 


disk_pred = parameter_mean[2] + 1.0875*(radial_SB/parameter_mean[1])

#plotting the curves for testing
fig,axes = plt.subplots()
axes.scatter(radial_SB_mod,B_band_mod,color = 'y', label = "Data")
plt.plot(radial_SB,disk_pred,color = 'k' , label = "model fit")
axes.set_ylabel("Surface Brightness (mag / arcsecond^2) ")
axes.set_xlabel("Radial Distance (Kpc)")
axes.set_ylim(24,18)
plt.legend()
plt.show()




R_d = 0.968
mu_0_disk = 19.755
I_0_disk = 790.6786279998262