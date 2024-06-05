import pandas as pd
import numpy as np
import scipy as sp
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#Constant Normalization
aa = .306601 #parsec/lightyear

#normalize the data and create a csv file for it 
AstroData = {'Star': ['Proxima Centauri', 'Sirius A', 'Betelgeuse', 'Vega', 'Rigel', 'Alpha Centauri A', 'Deneb', 'Arcturus', 'Capella Aa', 'Spica'], 
             'Type': ['Red Dwarf', 'Main Sequence', 'Red Supergiant', 'Main Sequence', 'Blue Supergiant', 'Main Sequence', 'Blue Supergiant', 'Red Giant', 'Yellow Giant', 'Binary System'],
             'Mass (M☉)': [.12, 2.02, 11.6, 2.1, 17.0, 1.1, 19.0, 1.08, 2.57, 10.25],
             'Luminosity (L☉)': [.0017, 25.4, 126000, 40.12, 120000, 1.519, 196000, 170, 78.7, 12100],
             'Temperature (K)': [3042, 9940, 3500, 9602, 12100, 5790, 8525, 4286, 4940, 25400],
             'Distance (Pc)': [4.24*aa, 8.6*aa, 642*aa, 25.05*aa, 860*aa, 4.37*aa, 2600*aa, 36.7*aa, 42.9*aa, 250*aa]}
astrodf = pd.DataFrame(AstroData)

# !AstronomyData.csv was created!
astrodf.to_csv('AstronomyData.csv', index = False)
# !AstronomyData.csv was created!

df = pd.read_csv("AstronomyData.csv")

#we use capital X for an independent dataset and lower case y for dependent data
X = df[['Mass (M☉)', 'Distance (Pc)']]
y = df[['Luminosity (L☉)', 'Temperature (K)']]

#sklearn's linear regression for a quick estimate on values and correlation
regres = linear_model.LinearRegression()
regres.fit(X,y)

#3rd order model functions using numpy 
MassLuminosityModel = np.poly1d(np.polyfit(df['Mass (M☉)'], df['Luminosity (L☉)'], 3))
LuminosityDistanceModel = np.poly1d(np.polyfit(df['Luminosity (L☉)'], df['Distance (Pc)'], 3))

#line for Mass and Luminosity Model, so the bounds will be on x axis from .12 to 10.25
#then we can do the same for Luminosity, Distance and create functions that plot the data and models
def MLplot():
    lineM = np.linspace(0,20,10)
    plt.scatter(df['Mass (M☉)'], df['Luminosity (L☉)'])
    plt.plot(lineM, MassLuminosityModel(lineM))
    plt.show()

def LDplot():
    lineL = np.linspace(0,200000, 20000)
    plt.scatter(df['Luminosity (L☉)'], df['Distance (Pc)'])
    plt.plot(lineL, LuminosityDistanceModel(lineL))
    plt.show()

#List for appending individual function values 
MLpoints = []
LDpoints = []
for x in df['Mass (M☉)']:
    MLpoints.append(MassLuminosityModel(x))
for x in df['Luminosity (L☉)']:
    LDpoints.append(LuminosityDistanceModel(x))

#find the r^2 value for each model by comparing function values and data values
relationshipML = r2_score(MLpoints, df['Luminosity (L☉)'])
relationshipLD = r2_score(LDpoints, df['Distance (Pc)'])