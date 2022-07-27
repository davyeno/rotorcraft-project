# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:28:05 2022

@author: csmit
"""
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def PhysHoverMod(filename): 

    data=pd.read_csv(filename+'.csv', parse_dates={'datetime':['date','time']}, 
                     index_col = 'datetime', low_memory = False)
    df = data
    #df.absoluteAltitude.plot()
    #plt.xlabel('Time')
    #plt.ylabel('Altitude')
    #plt.title('Altitude Graph of flight')
    #plt.show()

#Feature Engineering 
    #Define a function to identify turn direction    
    def CompassTurn(x):
        if x[-1] < 90 and x[0] > 270: 
            return ((x[-1]+360)-x[0])
        else: 
            return(x[-1]-x[0])
    #Use Rolling windows so each data point can look forward 
    df['turnDirection'] = df['trueHeading'].rolling(window = '5s').apply(CompassTurn)
    df['Altdiff'] = df['absoluteAltitude'].rolling(window = '20s').apply(lambda x: x[0]- x[-1], raw = True)
    df['longDiff'] = df['longitude'].rolling(window = '10s').apply(lambda x: x[-1]-x[0], raw=True)
    df['latDiff'] = df['latitude'].rolling(window = '10s').apply(lambda x: x[-1]-x[0], raw=True)
    
    # Maneuver Parameters 
    HoverLLimit = 2
    HoverULimit = 50
    StatHoverGroundSpeed = 5
    HoverAltInd = 12
    PedalTurnHeadingInd = 10   
    TABPInd = 0.00002
    
#Define a function for each Maneuver    
    def hover(x): 
        if x['absoluteAltitude'] > HoverLLimit and \
            x['absoluteAltitude'] < HoverULimit and \
            x['groundSpeed'] < StatHoverGroundSpeed and \
            x['Altdiff'] < HoverAltInd and \
            x['Altdiff'] > -HoverAltInd: return 1
        else: return 0 

    def LTurn(x): 
        if x['absoluteAltitude'] > HoverLLimit and \
            x['absoluteAltitude'] < HoverULimit and \
            x['groundSpeed'] < 2 and \
            x['Altdiff'] < HoverAltInd and \
            x['Altdiff'] > -HoverAltInd and \
            x['turnDirection'] < -PedalTurnHeadingInd: return 1
        else: return 0
    
    def RTurn(x): 
        if x['absoluteAltitude'] > HoverLLimit and \
            x['absoluteAltitude'] < HoverULimit and \
            x['groundSpeed'] < 2 and \
            x['Altdiff'] < HoverAltInd and \
            x['Altdiff'] > -HoverAltInd and \
            x['turnDirection'] > PedalTurnHeadingInd: return 1
        else: return 0 
    
    def TABPointCW(x): 
        if x['absoluteAltitude'] > HoverLLimit and \
            x['absoluteAltitude'] < HoverULimit and \
            x['groundSpeed'] < 5 and \
            x['groundSpeed'] > 2 and \
            x['Altdiff'] < HoverAltInd and \
            x['Altdiff'] > -HoverAltInd and \
            x['turnDirection'] > 15 and\
            x['longDiff'] > -0.0006 and \
            x['longDiff'] < 0.0002 and \
            x['latDiff'] < 0.0006 and \
            x['latDiff'] > -0.0006: return 1
        else: return 0
        
      
    def TABPointCCW(x): 
        if x['absoluteAltitude'] > HoverLLimit and \
            x['absoluteAltitude'] < HoverULimit and \
            x['groundSpeed'] < 5 and \
            x['groundSpeed'] > 2 and \
            x['Altdiff'] < HoverAltInd and \
            x['Altdiff'] > -HoverAltInd and \
            x['turnDirection'] < -15 and\
            x['longDiff'] > -0.0006 and \
            x['longDiff'] < 0.0002 and \
            x['latDiff'] < 0.0006 and \
            x['latDiff'] > -0.0006: return 1
        else: return 0
#Apply Functions to predict Maneuvers    
    df['LabHoverOrNot'] = df.apply(hover, axis = 1)
    df['LabRightTurn'] = df.apply(RTurn, axis = 1)
    df['LabLeftTurn'] = df.apply(LTurn, axis = 1)
    df['LabTurnABPCW'] = df.apply(TABPointCW, axis = 1)
    df['LabTurnABPCCW'] = df.apply(TABPointCCW, axis = 1) 
    
#Apply Rolling Means to the predicted data to smooth outliers 
    #Hover
    df['HoverMean']=df['LabHoverOrNot'].rolling(window = '17s').mean()
    df.loc[df['HoverMean']> 0.6, ['LabHoverOrNot']] = 1

    #Right Turn 
    df['RMean']=df['LabRightTurn'].rolling(window = '17s').mean()  
    df.loc[df['RMean']> 0.6, ['LabRightTurn']] = 1
    
    #Left Turn 
    df['LMean']=df['LabLeftTurn'].rolling(window = '17s').mean()
    df.loc[df['LMean']> 0.6, ['LabLeftTurn']] = 1
    
    #TABPCW
    df['CWMean']=df['LabTurnABPCW'].rolling(window = '17s').mean()
    df.loc[df['CWMean']> 0.6, ['LabTurnABPCW']] = 1
    
    #TABPCCW
    df['CCWMean']=df['LabTurnABPCCW'].rolling(window = '17s').mean()   
    df.loc[df['CCWMean']> 0.6, ['LabTurnABPCCW']] = 1



#Get Maneuver Start and Stop Times  
   
    #Hover    
    first_index = []
    arr = []
    for j,i in enumerate(df['LabHoverOrNot']):
        if j == 0:
            first_index.append([i,j])
        else:
            if i != df['LabHoverOrNot'][j-1]:
                first_index.append([df['LabHoverOrNot'][j-1],j-1])
                first_index.append([i,j])
    
    
    for i,j in enumerate(first_index):
        #print(i)
        #print(j)
        if j[0]!= 0 and first_index[i-1][0]==0 :
            arr.append(dict(Task="Hover %s"%i,
                            Start=df.index[first_index[i][1]],
                            Finish=df.index[first_index[i+1][1]],
                            Maneuver='Hover'
                           )
                      )
    
    ###RIGHT TURN 
    
    first_indexR = []
    
    for j,i in enumerate(df['LabRightTurn']):
        if j == 0:
            first_indexR.append([i,j])
        else:
            if i != df['LabRightTurn'][j-1]:
                first_indexR.append([df['LabRightTurn'][j-1],j-1])
                first_indexR.append([i,j])
    
    for i,j in enumerate(first_indexR):
        #print(i)
        #print(j)
        if j[0]!= 0 and first_indexR[i-1][0]==0 :
            arr.append(dict(Task="Right Turn %s"%i,
                            Start=df.index[first_indexR[i][1]],
                            Finish=df.index[first_indexR[i+1][1]],
                            Maneuver='Right Turn'
                           )
                      )
    
    ### LEFT TURN        
    first_indexL = []
    
    for j,i in enumerate(df['LabLeftTurn']):
        if j == 0:
            first_indexL.append([i,j])
        else:
            if i != df['LabLeftTurn'][j-1]:
                first_indexL.append([df['LabLeftTurn'][j-1],j-1])
                first_indexL.append([i,j])
    
    for i,j in enumerate(first_indexL):
        #print(i)
        #print(j)
        if j[0]!= 0 and first_indexL[i-1][0]==0:
            arr.append(dict(Task="Left Turn %s"%i,
                            Start=df.index[first_indexL[i][1]],
                            Finish=df.index[first_indexL[i+1][1]],
                            Maneuver='Left Turn'
                           )
                      )
    
    #TURN ABOUT A POINT CLOCKWISE       
    first_indexTABPCW = []
    
    for j,i in enumerate(df['LabTurnABPCW']):
        if j == 0:
            first_indexTABPCW.append([i,j])
        else:
            if i != df['LabTurnABPCW'][j-1]:
                first_indexTABPCW.append([df['LabTurnABPCW'][j-1],j-1])
                first_indexTABPCW.append([i,j])
    
    for i,j in enumerate(first_indexTABPCW):
        #print(i)
        #print(j)
        if j[0]!= 0 and first_indexTABPCW[i-1][0] == 0 :
            arr.append(dict(Task="TABP Clockwise %s"%i,
                            Start=df.index[first_indexTABPCW[i][1]],
                            Finish=df.index[first_indexTABPCW[i+1][1]],
                            Maneuver='TABP Clockwise'
                           )
                      )
    
    #Turn About a Point Counter-Clockwise         
    first_indexTABPCCW = []
    
    for j,i in enumerate(df['LabTurnABPCCW']):
        if j == 0:
            first_indexTABPCCW.append([i,j])
        else:
            if i != df['LabTurnABPCCW'][j-1]:
                first_indexTABPCCW.append([df['LabTurnABPCCW'][j-1],j-1])
                first_indexTABPCCW.append([i,j])
    
    for i,j in enumerate(first_indexTABPCCW):
        #print(i)
        #print(j)
        if j[0]!= 0 and first_indexTABPCCW[i-1][0] ==0 :
            arr.append(dict(Task="TABP Counter-Clockwise %s"%i,
                            Start=df.index[first_indexTABPCCW[i][1]],
                            Finish=df.index[first_indexTABPCCW[i+1][1]],
                            Maneuver='TABP Counter-Clockwise'
                           )
                      )
    
#Visualizations 
    Hoverdf = pd.DataFrame(arr)

    fig = px.timeline(Hoverdf, x_start="Start", x_end="Finish", 
                      y="Maneuver", color="Maneuver")
    fig.update_layout(title = {'text': filename + ' Hover Maneuver', 'x': 0.5})
    
    fig.show()

    
    fig = px.scatter_mapbox(df, lat = 'latitude', lon = 'longitude', 
                        color = 'LabHoverOrNot', zoom = 15, height = 600)
    fig.update_layout(mapbox_style = "open-street-map")
    fig.update_layout(margin = {"r": 0, "t": 30, "l": 0, "b": 0})
    fig.update_layout(title = {'text': "Grid Coordinates of Hover Maneuver",
                               'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'
                     })
    fig.show()

    
    
#Merge labels to Original dataframe and write to CSV
    labels = df[['LabHoverOrNot', 'LabRightTurn', 'LabLeftTurn', 
                 'LabTurnABPCW', 'LabTurnABPCCW']]
    data = pd.concat([data, labels], axis = 1)
    data.to_csv(filename+'_HoverLabels.csv', index_label = 'datetime')
    
    
    return(df, Hoverdf)