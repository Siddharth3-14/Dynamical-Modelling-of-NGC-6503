import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt

Data = pd.read_csv('../Data/velocity_dispersion.csv', delimiter = '\t',header = 'infer')
Radial_distance = np.array(Data["Radial Distance"])
velocity_dispersion = np.array(Data["Radial Velocity Dispersion"])

R_d = 0.789

model = pm.Model()

with model:
	#priors 
	sigma = pm.HalfNormal("sigma", sigma = 0.4)
	sigma_0 = pm.Gamma("sigma_0", alpha = 3, beta = 1)
	A = pm.Gamma("A", alpha = 3, beta = 1)
	B = pm.Gamma("B", alpha = 3, beta = 1)

	radial_dispersion = sigma_0*tt.exp(-1*Radial_distance/R_d)*(1 - A*tt.exp(-((Radial_distance**2)/(2*B*B)) ))


	#likelihood
	Y_obs = pm.Normal('Y_obs', mu = radial_dispersion, sigma = sigma, observed = velocity_dispersion)
	step = pm.Metropolis()
	trace = pm.sample(draws = 50000,step = step, tune = 2000, cores = 2)


parameter_mean = pm.summary(trace)["mean"]

velocity_dispersion_pred = parameter_mean[1]*np.exp(-1*Radial_distance/R_d)*(1 - parameter_mean[2]*np.exp(-((Radial_distance**2)/(2*parameter_mean[3]*parameter_mean[3])) ))


#plotting the curves for testing
fig,axes = plt.subplots()
#axes.scatter(radial_gas,H_beta_rot)
axes.scatter(Radial_distance,velocity_dispersion)
plt.plot(Radial_distance,velocity_dispersion_pred)
plt.show()