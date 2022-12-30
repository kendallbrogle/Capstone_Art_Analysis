#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 20:21:13 2022

@author: Kendall
"""
import numpy as np
import pandas as pd
import statistics 
import matplotlib.pyplot as plt 
from scipy import stats
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

random.seed(12063975)
seed = np.round(random.random()*100)

art = pd.read_csv('theArt.csv', delimiter = ',', usecols = np.arange(0,8))
data = pd.read_csv('theData.csv', delimiter = ',', header = None)

# Question 1
classical = data.iloc[:,:35]
clasnp = classical.to_numpy() 
clas_median = np.median(clasnp)
print(clas_median)
clasflat = clasnp.flatten() # 


modern = data.iloc[:,35:70]
modnp = modern.to_numpy() 
mod_median = np.median(modnp) 
print(mod_median)
modflat = modnp.flatten() 

u1, p1 = stats.mannwhitneyu(clasflat,modflat)
print(p1)

fig1, ax1 = plt.subplots()

bp1 = ax1.boxplot([clasflat,modflat],
                patch_artist=True,
                whiskerprops={'linewidth': 2, 'color': 'green'},  
                medianprops={'color': 'pink', 'linewidth': 2},  
                boxprops={'color': 'green', 'facecolor': 'lightblue'}) 

plt.title('Classical v Modern')
plt.xlabel('Group')
plt.ylabel('Value')
plt.show()

#Question 2 
nonhuman = data.iloc[:,70:91]
nonhnp = nonhuman.to_numpy()
nonh_median = np.median(nonhnp)
print(nonh_median)
nonhflat = nonhnp.flatten()

u2, p2 = stats.mannwhitneyu(nonhflat,modflat)
print(p2)

fig2, ax2 = plt.subplots()

bp2 = ax2.boxplot([nonhflat,modflat],
                patch_artist=True, 
                whiskerprops={'linewidth': 2, 'color': 'green'},  
                medianprops={'color': 'lightblue', 'linewidth': 2},  
                boxprops={'color': 'green', 'facecolor': 'pink'}) 

plt.title('NonHuman v Modern')
plt.xlabel('Group')
plt.ylabel('Value')
plt.show()

#Question 3 
#Column 217 is gender

menrow = data.loc[data[216]==1]
menrates = menrow.iloc[:,:91]
mennp = menrates.to_numpy()
menmedian = np.median(mennp)
menflat = mennp.flatten()
print(menmedian )
womenrow = data.loc[data[216]==2]
womenrates = womenrow.iloc[:,:91]
womennp = womenrates.to_numpy()
womenmedian = np.median(womennp)
print(womenmedian )
womenflat = womennp.flatten()

u3, p3 = stats.mannwhitneyu(menflat,womenflat)
print("Q 3")
print(p3)

fig3, ax3 = plt.subplots()

bp3 = ax3.boxplot([menflat,womenflat],
                patch_artist=True,
                whiskerprops={'linewidth': 2, 'color': 'blue'},  
                medianprops={'color': 'yellow', 'linewidth': 2},  
                boxprops={'color': 'blue', 'facecolor': 'purple'}) 

plt.title('Men v Women')
plt.xlabel('Group')
plt.ylabel('Value')
plt.show()

#Question 4
#Column 219 is art

noartback = data.loc[data[218]== 0]
nab = noartback.iloc[:,:91]
nab_np = nab.to_numpy()
nabflat = nab_np.flatten()

artback = data.loc[data[218].isin([1,2,3])]
ab = artback.iloc[:,:91]
ab_np = ab.to_numpy()
abflat = ab_np.flatten()

u4, p4 = stats.mannwhitneyu(nabflat,abflat)
print(p4)

fig4, ax4 = plt.subplots()
bp4 = ax4.boxplot([nabflat,abflat],
                patch_artist=True,
                whiskerprops={'linewidth': 2, 'color': 'pink'},  
                medianprops={'color': 'yellow', 'linewidth': 2},  
                boxprops={'color': 'blue', 'facecolor': 'lightgreen'}) 

plt.title('Art Experience')
plt.xlabel('Group')
plt.ylabel('Value')
plt.show()

#Question 5
#Columns 92-182 are energy of paintings

energy = data.iloc[:,91:182]
energy = energy.dropna()

rating = data.iloc[:,:91]
model5 = LinearRegression()
model5.fit(energy,rating)

kfold5 = KFold(n_splits= 3, random_state=int(seed), shuffle=True) 

scores5 = cross_val_score(model5, energy, rating, cv = kfold5)
mean5 = scores5.mean()
std5 = scores5.std()
pred5 = model5.predict(energy)
print("Number 5:")
print(scores5)
print(mean5)
print(std5)
r25 = r2_score(rating, pred5)
print(r25)

mse5 = mean_squared_error(rating, pred5)
rmse5 = np.sqrt(mse5)
print("RMSE q 5 ")
print(rmse5)

#plt.scatter(energy,rating)
plt.plot(range(1, 4), scores5, linestyle='--', marker='o', label='Cross-validation scores')

plt.scatter(rating, pred5, color = 'pink' )
plt.title('Prefrence Predicted from Energy')
plt.xlabel('Actual ratings')
plt.ylabel('Predicted ratings')
plt.show()

#Question 6
demographic = data.iloc[:,215:221]
demo = demographic.dropna()
demo_energy = pd.concat([demo,energy], axis =1)
x6 = demo_energy.dropna()
y6 = data.iloc[:279,:91]

zscore = stats.zscore(x6)

kfold = KFold(n_splits= 7, random_state=int(seed), shuffle=True) 

model6 = LinearRegression()
model6.fit(x6,y6)
scores6 = cross_val_score(model6, zscore, y6, scoring='r2',
                         cv=kfold, n_jobs=-1)
pred6 = model6.predict(x6)

print(np.abs(scores6))
print("avg")
avg6 = np.mean(scores6)
print(avg6)
plt.title('Cross Scores')
plt.xlabel('Score Number')
plt.ylabel('Score Value')
plt.plot(np.abs(scores6),color = 'green' )
plt.show()
rmse_scores6 = np.sqrt(-scores6)
mean_rmse = rmse_scores6.mean()
print(mean_rmse)

r26 = r2_score(y6, pred6)
print("R2 for 6")
print(r26)

plt.scatter(y6, pred6, color = 'green' )
plt.title('Prefrence Predicted from Energy and Demographic')
plt.xlabel('Actual ratings')
plt.ylabel('Predicted ratings')
plt.show()

#Question 7
energy_mean = energy.mean(axis = 1)
rating_mean = rating.mean(axis = 1)


rate_energy = np.column_stack((rating_mean, energy_mean ))
kmeans = KMeans(n_clusters=3).fit(rate_energy)
labels = kmeans.predict(rate_energy)
centers = kmeans.cluster_centers_

score = silhouette_score(rate_energy, labels)
print("Question 7: sil score")
print(score)

plt.scatter(rate_energy[:,0], rate_energy[:,1], c=labels, cmap = 'gist_rainbow')
plt.title('K Means Clusters: Energy and Pref')
plt.xlabel('Mean Pref Rating')
plt.ylabel('Mean Energy Rating')
plt.show()

modern = data.iloc[:,35:70]
rowmeans = modern.mean()
modernmean = rowmeans.mean()

#Question 8
self = data.iloc[:,205:215]
self = self.dropna()
r = data.iloc[:286,:91]

pca8 = PCA(n_components=1)
x8 = pca8.fit_transform(self)

x_train8, x_test8, y_train8, y_test8 = train_test_split(x8, r, test_size=0.2,random_state= int(seed))

model8 = LinearRegression()
model8.fit(x_train8, y_train8)

loadings8 = pca8.components_.T
explained_variance8 = pca8.explained_variance_
plt.scatter(range(len(loadings8)), loadings8, color = 'pink')
#plt.plot(range(len(loadings8)), loadings8, '-o', color='pink')
plt.title('Loading PCA: Self Image')
plt.xlabel('Feature index')
plt.ylabel('Loading')
plt.show()

y_pred8 = model8.predict(x_test8)
mse8 = mean_squared_error(y_test8, y_pred8)
rmse8= np.sqrt(mse8)
r2_8 = r2_score(y_test8, y_pred8)

print("Q 8 R2:")
print(r2_8)
print("Q 8 RMSE:")
print(rmse8)

loadings8 = pca8.components_
features8 = []
for component in loadings8:
  feature8 = abs(component).argmax()
  features8.append(feature8)

print(features8)

#Question 9
dark = data.iloc[:,182:194].dropna()
y9 = data.iloc[:284,:91]

#Determine Number of components
pca9 = PCA()
pca9.fit(dark)
eigenvalues = pca9.explained_variance_

plt.plot(range(1, len(eigenvalues)+1), eigenvalues, 'o-', color = 'green')
plt.title('PCA Components: Dark Personality')
plt.xlabel('Component')
plt.ylabel('Eigenvalue')
plt.show()

pca1 = PCA(n_components=1)
x_pca1 = pca1.fit_transform(dark)

pca2 = PCA(n_components=2)
x_pca2 = pca2.fit_transform(dark)

pca3 = PCA(n_components=3)
x_pca3 = pca3.fit_transform(dark)

kf9 = KFold(n_splits=10, random_state= int(seed), shuffle=True)

x_train9, x_test9, y_train9, y_test9 = train_test_split(x_pca2, y9, test_size=0.2, random_state= int(seed))
model9 = LinearRegression()
model9.fit(x_train9, y_train9)

y_pred9 = model9.predict(x_test9)
mse9 = mean_squared_error(y_test9, y_pred9)
rmse9= np.sqrt(mse9)
r2_9 = r2_score(y_test9, y_pred9)

print("Question 9 R2:")
print(r2_9)
print("Question 9 RMSE:")
print(rmse9)

loadings9 = pca3.components_
features = []
for component in loadings9:
  feature = abs(component).argmax()
  features.append(feature)

print(features)

#Question 10
X10 = data.drop(217, axis = 1)
x10 = X10.dropna()
y10 = data.iloc[:296,217].dropna()
y10 = y10.to_numpy()

b = [-float('inf'), 2, float('inf')]

#y10 = y10.values.reshape(1,-1)
y10_binned = pd.cut(y10.flatten(), bins = b , labels = [0,1])
#y10_binned = y10_binned.values

x_train10, x_test10, y_train10, y_test10 = train_test_split(x10, y10_binned, test_size=0.2, random_state=int(seed))
clf10 = RandomForestClassifier(n_estimators=100)
clf10.fit(x_train10, y_train10)
y_pred10 = clf10.predict(x_test10)
accuracy = clf10.score(x_test10, y_test10)
print("Question 10:")
print(accuracy)

confusion_matrix = confusion_matrix(y_test10, y_pred10)
sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.title('Polical Orientation Pred')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

print(confusion_matrix)

#Extra Credit 
action = data.iloc[:,194:205].dropna()
yE = data.iloc[:285,:91]

#Determine Number of components
pcaE = PCA()
pcaE.fit(action)
eigenvaluesE = pcaE.explained_variance_

plt.plot(range(1, len(eigenvaluesE)+1), eigenvaluesE, 'o-', color = 'purple')
plt.title('PCA Components: Action Scores')
plt.xlabel('Component')
plt.ylabel('Eigenvalue')
plt.show()

pca1E = PCA(n_components=1)
x_pca1E = pca1E.fit_transform(action)

pca2E = PCA(n_components=2)
x_pca2E = pca2E.fit_transform(action)

pca5E = PCA(n_components=5)
x_pca5E = pca5E.fit_transform(action)

kfE = KFold(n_splits=10, random_state= int(seed), shuffle=True)

x_trainE, x_testE, y_trainE, y_testE = train_test_split(x_pca5E, yE, test_size=0.2, random_state= int(seed))
modelE = LinearRegression()
modelE.fit(x_trainE, y_trainE)

y_predE = modelE.predict(x_testE)
mseE = mean_squared_error(y_testE, y_predE)
rmseE= np.sqrt(mseE)
r2_E = r2_score(y_testE, y_predE)

print("E R2:")
print(r2_E)
print("E RMSE:")
print(rmseE)

loadingsE = pca5E.components_
featuresE = []
for component in loadingsE:
  featureE = abs(component).argmax()
  featuresE.append(featureE)

print(featuresE)




