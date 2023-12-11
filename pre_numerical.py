# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 22:24:39 2023

@author: Leander
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


r = np.load('r.npy') * 1000 #load radius contour
z = 1000*np.load('z.npy') + 15.1#load z coords
hc=np.load('hc.npy') * 1000 #load channel height and conv to mm
h = np.load('h.npy')*1000 # load gas side wall thickness in mm
a = np.load('a.npy')*1000 #load coolant channel width in mm
#r= r+h #add wall thickness to get radius of coolant side wall 

drdz = np.diff(r)/np.diff(z)
abs_n = np.hypot(drdz, np.diff(z))
norm_nx = drdz / abs_n
norm_ny = np.diff(z) / abs_n
norm_nx = np.insert(norm_nx, 0, norm_nx[0])
norm_ny = np.insert(norm_ny, 0, norm_ny[0])
z = z - norm_nx * h
r = r + norm_ny * h

a0 = a[0] #get coolant channel radius at cc
r0= r[0] #get cc radius

phi = [0.03*i for i in range(len(r))] #baseline phi
############################################

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def Vangle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

x = [r[i] * np.sin(angle) for i,angle in enumerate(phi)]
y = [r[i] * np.cos(angle) for i,angle in enumerate(phi)]

def channel_points(x_p,y_p, hc):
    #hc = np.interp(r, [r_list.min(),r_list.max()], [h0-hct,h0])
    hc = hc
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





df_big = pd.DataFrame(columns=["x1","x2","x3","x4","y1","y2","y3","y4","z"])

for i in range(len(x)):
    temp = channel_points(x[i],y[i],hc[i])
    temp.append(z[i])
    df_big.loc[i,:] = temp



def calculate_angle(i): #calculates angle in rad between element i and the next one 
    x1 =df_big.iloc[i][0:4].mean()
    x2 =df_big.iloc[i+1][0:4].mean()
    y1 =df_big.iloc[i][4:8].mean()
    y2 = df_big.iloc[i+1][4:8].mean()
    z1 = df_big.iloc[i][-1]
    z2 = df_big.iloc[i+1][-1]
    angle = Vangle([x2-x1,y2-y1,z2-z1],[0,0,1])
    return angle

def target_function(i):
    OUTPUT = min(35, i * 4 - 20, (i-98) * -2+15)
    if OUTPUT < 0:
        OUTPUT = 0
    return OUTPUT
target_function = np.vectorize(target_function)
#plt.plot(np.linspace(0, 99, 100), target_function(np.linspace(0, 99, 100)))


r_pt_phc0 = r + hc
r_pt_phc1 = [(row[2]**2+row[6]**2)**0.5 for _,row in df_big.iterrows()]
    

#### iterative solver start
for i in range(0,99):
    delta_base = 0.00005
    angle = calculate_angle(i)
    angle = angle*180/np.pi
    target_angle = target_function(i)
    error = target_angle - angle
    #print(error)
    n = 0
    while abs(error) > 1E-4 and n<25000:
        if abs(error) > 10: 
            delta = delta_base * 20
        elif abs(error) > 5:
            delta = delta_base * 5
        elif abs(error) > 1:
            delta = delta_base
        elif abs(error) > .01:
            delta = delta_base /100
        elif abs(error) > .001:
            delta = delta_base /1000
        elif abs(error) > .0001:
            delta = delta_base /10000
        else: 
            delta = delta_base / 500000
            
        if error > 0:
            phi[i+1] = phi[i+1] + delta
        else:
            phi[i+1] = phi[i+1] - delta
            
        x_new = r[i+1] * np.sin(phi[i+1])
        y_new = r[i+1] * np.cos(phi[i+1])
        df_big.iloc[i+1] = [*channel_points(x_new,y_new,hc[i+1]),z[i+1]]
        angle = calculate_angle(i)
        error = target_angle - angle*180/np.pi
        #if n%350 == 0:
            #print("Step: {}, Substep: {}, Error {}, current Angle: {}, current Phi {}".format(i,n,error,angle*180/np.pi,phi[i+1]))
        n +=1
    print(str(i) + " / 99:" + " Target angle " + str(target_angle) + " Final angle " + str(angle * 180 / np.pi) )
        
##iterative solver ennd
#debug
centerx = []
centery = []
centerz = []
#recalc of center points
for _,row in df_big.iterrows():
    centerx.append(row[:4].mean())
    centery.append(row[4:8].mean())
    centerz.append(row[-1].mean())

norm = [0,0,1] #z axis

vectors = []
#getting vectors
for i in range(len(centerx)-1):
    pointx =  centerx[i+1] - centerx[i]
    pointy = centery[i+1] - centery[i]
    pointz =  centerz[i+1] - centerz[i]
    vectors.append([pointx,pointy,pointz])

angles = []

#calculating all the angles
def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

for vector in vectors:
    angles.append(angle(vector,norm))

angles = [angle * 180 / np.pi for angle in angles]
plt.plot(z[0:99],angles)
plt.show()
plt.plot(z,phi)

##debug end

#output coordinate csv 
df_big *=.1
df_big['phi'] = a0/r0
df_big.to_csv("surfacetest.csv", index=False, header=False)
    









