import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('../Data/SB.csv', delimiter = '\t',header = 'infer')
Y = data["B"]
X = data["Radial Distance"]

fig, ax = plt.subplots()
ax.scatter(X,Y)
ax.set_ylim(25,19)
plt.show()