import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt

#surface brightness
SB_data = pd.read_csv('../Data/SB.csv', delimiter = '\t',header = 'infer')

radial_SB = np.array(SB_data["Radial Distance"])
B_band = np.array(SB_data["B"])


total_SB = pm.Model()
with total_SB:
	#priors
	sigma = pm.HalfNormal("sigma" , sigma = 0.4)
	R_d = pm.Gamma("R_d" , alpha = 3, beta = 1, testval = 2.16)
	rb = pm.Gamma("rb" , alpha = 3, beta = 1)
	R_h = pm.Uniform("R_h", lower = 0.3, upper = 1.5)
	alpha = pm.Gamma("alpha" , alpha = 3, beta = 1)
	nb = pm.Gamma("nb" , alpha = 3, beta = 1)
	mu_0_bulge = pm.Gamma("mu_0_bulge" , alpha = 3, beta = 1)
	mu_0_disk = pm.Gamma("mu_0_disk" , alpha = 3, beta = 1)


	mu_before = mu_0_bulge + 1.0875*((radial_SB/rb)**(1/nb))
	mu_after = mu_0_disk + 1.0875*(radial_SB/R_d + ((R_h/radial_SB)**alpha))

	mu_total = pm.math.switch( R_h >= radial_SB, mu_before, mu_after)


	#likelihood
	Y_obs = pm.Normal('Y_obs', mu = mu_total, sigma = sigma, observed = B_band)
	step = pm.Metropolis() 
	trace = pm.sample(draws = 500000 ,step = step, tune = 2000, cores = 2)

print(pm.summary(trace))
parameter_mean = pm.summary(trace)["mean"]
print("central luminosty of bulge ", 10**((27-parameter_mean[6])/2.5))
print("central luminosty of disk ",10**((27-parameter_mean[7])/2.5)) 

model_pred = np.zeros(radial_SB.shape[0])
for i in range(radial_SB.shape[0]):
	if radial_SB[i] < parameter_mean[3]:
		model_pred[i] = parameter_mean[6] + 1.0875*((radial_SB[i]/parameter_mean[2])**(1/parameter_mean[5]))
	else:
		model_pred[i] = parameter_mean[7] + 1.0875*(radial_SB[i]/parameter_mean[1] + ((parameter_mean[3]/radial_SB[i])**parameter_mean[4]))

bulge_pred = parameter_mean[6]+ 1.0875*((radial_SB/parameter_mean[2])**(1/parameter_mean[5]))
disk_pred = parameter_mean[7] + 1.0875*(radial_SB/parameter_mean[1] + ((parameter_mean[3]/radial_SB)**parameter_mean[4]))


#plotting the curves for testing
fig,axes = plt.subplots()
axes.scatter(radial_SB,B_band,color = 'y', label = 'Data')
plt.plot(radial_SB,model_pred,color = 'k' , label = "model fit")
plt.plot(radial_SB,bulge_pred,color = 'r', label = 'bulge contribution', linestyle='dashed')
plt.plot(radial_SB,disk_pred,color = 'g', label = 'disk contribution', linestyle='dashed')
axes.set_ylabel("Surface Brightness (mag / arcsecond^2) ")
axes.set_xlabel("Radial Distance (Kpc)")
axes.set_ylim(24,18)
plt.legend()
plt.show()


# R_d = 0.759
# rb = 0.104
# R_h = 0.473
# alpha = 1.599
# nb = 2.900
# mu_0_bulge = 18.742
# mu_0_disk = 18.799
# I_0_bulge = 2010.018247113691
# I_0_disk = 1907.2165205939068






