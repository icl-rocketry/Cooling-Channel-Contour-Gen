#Author-
#Description-

import adsk.core, adsk.fusion, adsk.cam, traceback
import csv
import math
#import numpy as np
def run(context):
    ui = None
    try:
    
        app = adsk.core.Application.get()
        ui  = app.userInterface
        ui.messageBox('ChannelTest')
        design = app.activeProduct
        rootComp = design.rootComponent 

        
        ############# preprocessing ####################
        h0 = 2
        a0 = 4
        r0 = 50
        phi = a0/r0 #* 180/math.pi

        x1 = []
        x2 = []
        x3 = []
        x4 = []
        y1 = []
        y2 = []
        y3 = []
        y4 = []
        z = []

        with open('C:\\Users\\Leander\\Documents\\iclr\\F360Skripts\\CC-SurfaceGenerator\\surfacetest.csv', 'r') as file:
            my_reader = csv.reader(file, delimiter=',')
            for row in my_reader:
                x1.append(float(row[0]))
                x2.append(float(row[1]))
                x3.append(float(row[2]))
                x4.append(float(row[3]))
                y1.append(float(row[4]))
                y2.append(float(row[5]))
                y3.append(float(row[6]))
                y4.append(float(row[7]))
                z.append(float(row[8]))



        ###### adding splines ############
        sketches = rootComp.sketches
        sketch = sketches.add(rootComp.xYConstructionPlane)
        
        
        points = adsk.core.ObjectCollection.create()


        for i in range(len(x1)):    
            p1 = adsk.core.Point3D.create(x1[i],y1[i],z[i]) 
            p2 = adsk.core.Point3D.create(x2[i],y2[i],z[i])
            p3 = adsk.core.Point3D.create(x3[i],y3[i],z[i])
            p4 = adsk.core.Point3D.create(x4[i],y4[i],z[i])
            arccenterpoint = adsk.core.Point3D.create(0,0,z[i])

            lines = sketch.sketchCurves.sketchLines
            line1 = lines.addByTwoPoints(p1,p3)
            line2 = lines.addByTwoPoints(p2,p4)
            
            
            arcs = sketch.sketchCurves.sketchArcs
            arc1 = arcs.addByCenterStartSweep(arccenterpoint,p1,-phi)
            arc2 = arcs.addByCenterStartSweep(arccenterpoint,p3,-phi)
        
        #lofting
        loftFeats = rootComp.features.loftFeatures
        loftInput = loftFeats.createInput(adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        loftSectionsObj = loftInput.loftSections
        for i in range(len(x1)):
            loftSectionsObj.add(sketch.profiles.item(i))
        loftInput.isSolid = True
        loftInput.isClosed = False
        loftInput.isTangentEndgesMerged = True
        
        loftFeats.add(loftInput)
        
    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))