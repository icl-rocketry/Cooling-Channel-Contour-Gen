# CC-geom-Generator
 generates a single helical cooling channel geometry in fusion360 given an r,z, and hc array

## Installation 

1. add the following to the enginesizer code: 

```python
hc = thanos_channel.hc
x = thanos_contour.x
r = thanos_contour.r
np.save('r',r)
np.save('z',x)
np.save('hc',hc)
```
2. after running the enginesizercode copy the files into the CC-geom-Generator folder.
3. add the folder of CC-geom-Generator as a Fuision360 script
4. Run the preprocessor.py file
5. Run the script in Fuison 360

Note: If changes to the channel geometry are made in the enginesizer code the channel_points() function in preprocessing.py needs to be updated with the correct boundary conditions