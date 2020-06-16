import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt


I_0_bulge = 2010.018
I_0_disk = 1907.2165


def simpsons_integration(a,b,N,func):
    h = (b-a)/N
    Xl0 = func(a) + func(b)
    Xl1 = 0 #summation of odd terms
    Xl2 = 0 #summation of even terms
    for i in range(1,N):
        X = a + i*h
        if i%2 == 0:
            Xl2 = Xl2 + func(X)
        elif i%2 != 0:
            Xl1 = Xl1 + func(X)
    Xl = h*(Xl0 + 2*Xl2 + 4*Xl1)/3.0
    return Xl

def Bulge(R,rb = 0.104,nb = 2.900):
    temp = 4*I_0_bulge*np.pi*np.exp(- ((R/rb)**(1/nb) ))*R*R
    return temp

def Disc(R,R_d = 0.759, R_h = 0.473, alpha = 1.599):
    temp = 4*I_0_disk*np.pi*0.24*np.exp(-(R/R_d + (R/R_h)**(-alpha)))*R
    return temp

#rotational curve

modelling_data = pd.read_csv("../Data/modelling_data.csv",delimiter = '\t',header = 'infer')
Radial_distance = np.array(modelling_data["Radial Distance"])


V_obs = np.array(modelling_data["V_obs"])
V_gas = np.array(modelling_data["V_gas"])
V_disk = np.array(modelling_data["V_disk"])

V_obs2 = (V_obs)*(V_obs)
V_gas2 = (V_gas)*(V_gas)
V_obs2 = V_obs2 - V_gas2

M_R_bulge = []
for i in Radial_distance:
    M_R_bulge.append(simpsons_integration(0.0001,i,5000,Bulge)/i)
M_R_bulge = np.array(M_R_bulge)

M_R_disk = []
for i in Radial_distance:
    M_R_disk.append(simpsons_integration(0.0001,i,5000,Disc)/i)
M_R_disk = np.array(M_R_disk)




total_model = pm.Model()

with total_model:

    #priors
    sigma = pm.HalfNormal("sigma" , sigma = 0.4)
    gamma = pm.Gamma("gamma", alpha = 3, beta = 1)
    ah = pm.Gamma("ah", alpha = 3, beta = 1)
    Mh = pm.Gamma("Mh", alpha = 3, beta = 1)
    M_by_L_bulge =  pm.Gamma("M_by_L_bulge", alpha = 3, beta = 1)
    M_by_L_disk = pm.Gamma("M_by_L_disc", alpha = 3, beta = 1)

    bulge_rot = M_by_L_bulge*M_R_bulge
    disk_rot = M_by_L_disk*M_R_disk
    halo_rot = (Mh*Radial_distance**(gamma - 1))/((ah**gamma)*(1 + ((Radial_distance/ah)**(gamma-1)) ))
    total_rot =  bulge_rot + disk_rot + halo_rot 

    #likelihood
    Y_obs = pm.Normal('Y_obs', mu = total_rot, sigma = sigma, observed = V_obs2)
    step = pm.Metropolis()
    trace = pm.sample(draws = 1000000, step = step, tune = 1000, cores = 2)



print(pm.summary(trace))
parameter_mean = pm.summary(trace)["mean"]
model_pred = (parameter_mean[4]*M_R_bulge + parameter_mean[5]*M_R_disk + (parameter_mean[3]*(Radial_distance**(parameter_mean[1] - 1)))/((parameter_mean[2]**parameter_mean[1])*(1 + ((Radial_distance/parameter_mean[2])**(parameter_mean[1]-1)) )))**0.5
bulge_pred = (parameter_mean[4]*M_R_bulge)**0.5
disk_pred = (parameter_mean[5]*M_R_disk)**0.5
halo_pred = ((parameter_mean[3]*(Radial_distance**(parameter_mean[1] - 1)))/((parameter_mean[2]**parameter_mean[1])*(1 + ((Radial_distance/parameter_mean[2])**(parameter_mean[1]-1)) )))**0.5


fig,axes = plt.subplots()
axes.scatter(Radial_distance,(V_obs2)**0.5,color = 'y' , label = 'Data')
plt.plot(Radial_distance,model_pred, color = 'k', label = 'Model Prediction')
plt.plot(Radial_distance,bulge_pred, color = 'r', label = 'bulge contribution')
plt.plot(Radial_distance,disk_pred, color = 'g', label = 'Disk contribution')
plt.plot(Radial_distance,halo_pred, color ='b', label = 'Halo contribution')
axes.set_ylabel("Rotational Velocity (Km/s) ")
axes.set_xlabel("Radial Distance (Kpc)")
plt.legend()
plt.show()











