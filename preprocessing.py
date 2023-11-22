# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 11:15:29 2023

@author: Leander
"""

import adsk.core, adsk.fusion, adsk.cam, traceback
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


r = np.load('r.npy')
r *= 1000
r += 1.5
z = 1000*np.load('z.npy')
hc=np.load('hc.npy') * 1000



#generating phi (kinda sketchy??)
#variant 1

dz = np.diff(z) #dz is constant so 
dz = dz[0]
dr= np.diff(r)
dr = np.insert(dr, 0,0)
tanphi = np.tan(40*np.pi/180)
a1 = (dz/tanphi)**2
rsquare = np.power(r,2)
drsquare = np.power(dr,2)
dt = 0.5*np.sqrt((a1-drsquare)/rsquare)
t1 = [sum(dt[0:i]) for i in range(len(dt))]


#variant 2

dz = np.diff(z) #dz is constant so 
dz = dz[0]
dr= np.diff(r)
dr = np.insert(dr, 0,0)
tanphi = np.tan(25*np.pi/180)
a2 = 1*(dz*tanphi)**2
rsquare = np.power(r,2)
drsquare = np.power(dr,2)
dt = np.sqrt(abs(a2-drsquare)/rsquare)
t2 = [sum(dt[0:i]) for i in range(len(dt))]

#variant 3
dz = np.diff(z) #dz is constant so 
dz = dz[0]
dr= np.diff(r)
dr = np.insert(dr, 0,0)
tanphi = np.tan((90-40)*np.pi/180)
rsquare = np.power(r,2)
drsquare = np.power(dr,2)
a1 = (dz/tanphi)**2 -drsquare
a2 = np.sqrt(abs(a1))
dt = 1*np.arcsin(a2/(2*r))

t3= [sum(dt[0:i]) for i in range(len(dt))]

dt = [2*np.pi/len(z) for i in range(len(z))]
t4 = [sum(dt[0:i]) for i in range(len(dt))]


#variant 4
#phi = np.linspace(0,2 * np.pi,100)
#x = [r[i] * np.sin(0.5*z[i]/(r[i])*np.tan(15*np.pi/180)) for i,angle in enumerate(phi)]
#y = [r[i] * np.cos(0.5*z[i]/(r[i])*np.tan(15*np.pi/180)) for i,angle in enumerate(phi)]

#variante 5
angle = 15
dthetabs = abs(np.diff(np.tan(angle*np.pi/180)*z/r))
dthetabs = np.insert(dthetabs, 0, dthetabs[0])
t6= [sum(dthetabs[0:i]) for i in range(len(dthetabs))]

### manual values for dtheta (small when dr/dz small, big when dr/dz big)
dz = np.diff(z) #dz is constant so 
dz = dz[0]
dr= np.diff(r)
dr = np.insert(dr, 0,0)
drodz = dr/dz
dt = []
for i in range(len(drodz)):
    base_val = 0.03
    if abs(drodz[i]) > 0.2:
        val = base_val*3
    else:
        val = base_val
    dt.append(val)

t7= [sum(dt[0:i]) for i in range(len(dt))]


#choose variant for phi (t3 = best??)
phi = t2
x = [r[i] * np.sin(angle) for i,angle in enumerate(phi)]
y = [r[i] * np.cos(angle) for i,angle in enumerate(phi)]



#z = [2*np.pi*r[i] * np.tan(40*np.pi/180)*angle for i,angle in enumerate(phi)]
ax = plt.figure().add_subplot(projection='3d')
ax.plot(x,y,z, 'o')

ax = plt.figure().add_subplot()
ax.plot(x[0],y[0], 'o')
plt.xlim(-60,60)
plt.ylim(-60,60)


def channel_points(x_p,y_p, hc):
    #hc = np.interp(r, [r_list.min(),r_list.max()], [h0-hct,h0])
    a0 = 4
    hc = hc
    r0 = 50
    deltaphi = a0/r0
    r = (x_p**2 + y_p**2)**0.5
    phi = np.arctan2(y_p,x_p)
    x1 = r * np.cos(phi)
    x2 = r * np.cos(phi-deltaphi)
    x3 = (r+hc) * np.cos(phi)
    x4 = (r+hc) * np.cos(phi-deltaphi)
    y1 = r * np.sin(phi)
    y2 = r * np.sin(phi-deltaphi)
    y3 = (r+hc) * np.sin(phi)
    y4 = (r+hc) * np.sin(phi-deltaphi)
    return [x1,x2,x3,x4,y1,y2,y3,y4]

#### create big csv:
#### with first colimn array of x, second column: array of y[], third column: z
df_big = pd.DataFrame(columns=["x1","x2","x3","x4","y1","y2","y3","y4","z"])

for i in range(len(x)):
    temp = channel_points(x[i],y[i],hc[i])
    temp.append(z[i])
    df_big.loc[i,:] = temp


#print(channel_points(x[0],y[0]))
#x_s, y_s = channel_points(x[0],y[0])
ax = plt.figure().add_subplot()
#ax.plot(x_s,y_s, 'o')
#ax.plot(x,y, 'o')
plt.xlim(-60,60)
plt.ylim(-60,60)
ax = plt.figure().add_subplot()
ax.plot(df_big['x1'],df_big['y1'], 'o')
ax.plot(df_big['x2'],df_big['y2'], 'o')
ax.plot(df_big['x3'],df_big['y3'], 'o')
ax.plot(df_big['x4'],df_big['y4'], 'o')
plt.xlim(-60,60)
plt.ylim(-60,60)


###### debug #######


x = df_big.iloc[0,0:4]
y = df_big.iloc[0,4:8]
ax = plt.figure().add_subplot()
ax.plot(x,y,'o')

ax = plt.figure().add_subplot()
ax.plot(z,t1, label="changing dt1")
ax.plot(z,t2, label="changing dt2")
ax.plot(z,t3, label="changing dt3")
ax.plot(z,t4, label="constant dt")
ax.plot(z,t6, label="weird theta absolute stuff idek anymore")
ax.plot(z,t7, label="manual whatever")
plt.legend()

centerx = []
centery = []
centerz = []
for _,row in df_big.iterrows():
    centerx.append(row[:4].mean())
    centery.append(row[4:8].mean())
    centerz.append(row[-1].mean())

norm = [0,0,1]

vectors = []

for i in range(len(centerx)-1):
    pointx =  centerx[i+1] - centerx[i]
    pointy = centery[i+1] - centery[i]
    pointz =  centerz[i+1] - centerz[i]
    vectors.append([pointx,pointy,pointz])

angles = []

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

for vector in vectors:
    angles.append(angle(vector,norm))

angles = [angle * 180 / np.pi for angle in angles]
print(max(angles))

### save coordinate csv
df_big *=.1
df_big.to_csv("surfacetest.csv", index=False, header=False)
