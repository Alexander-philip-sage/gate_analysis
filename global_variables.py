import os
RIGHT_AVY_HEADER = 'Angular Velocity Y (rad/s).2'
LEFT_AVY_HEADER = 'Angular Velocity Y (rad/s).3'
COLUMNS_TO_GRAPH = ['Acceleration Y (m/s^2)',  'Angular Velocity Y (rad/s)',##right thigh 
                    'Acceleration Y (m/s^2).1','Angular Velocity Y (rad/s).1',##left thigh
                    'Acceleration Y (m/s^2).2', RIGHT_AVY_HEADER,##right shank
                    'Acceleration Y (m/s^2).3',LEFT_AVY_HEADER ]##left shank
tmp = []
for val in COLUMNS_TO_GRAPH:
  tmp.append(val.replace('Y','X'))
  tmp.append(val.replace('Y','Z'))
COLUMNS_TO_GRAPH.extend(tmp)

COLUMNS_TO_AREA = {  'Acceleration Y (m/s^2)':'right thigh',  'Angular Velocity Y (rad/s)':'right thigh',##right thigh 
                    'Acceleration Y (m/s^2).1':'left thigh','Angular Velocity Y (rad/s).1':'left thigh',##left thigh
                    'Acceleration Y (m/s^2).2':'right shank', RIGHT_AVY_HEADER:'right shank',##right shank
                    'Acceleration Y (m/s^2).3':'left shank',LEFT_AVY_HEADER:'left shank' }##left shank     
tmp = list(COLUMNS_TO_AREA.items())
for val in tmp:
  COLUMNS_TO_AREA[val[0].replace('Y','X')] = val[1]
  COLUMNS_TO_AREA[val[0].replace('Y','Z')] = val[1]
COLUMNS_TO_LEG = {}
for k in list(COLUMNS_TO_AREA.keys()):
  COLUMNS_TO_LEG[k]=COLUMNS_TO_AREA[k].split(' ')[0]
COLUMNS_BY_SENSOR = [{'sensor':'thigh','right':'Acceleration Y (m/s^2)','left': 'Acceleration Y (m/s^2).1'}, 
                     {'sensor':'thigh','right':'Angular Velocity Y (rad/s)', 'left':'Angular Velocity Y (rad/s).1'},
                     {'sensor':'shank','right':'Acceleration Y (m/s^2).2','left':'Acceleration Y (m/s^2).3'},
                     {'sensor':'shank','right':RIGHT_AVY_HEADER, 'left':LEFT_AVY_HEADER},]    
tmp = COLUMNS_BY_SENSOR.copy()        
for d in tmp:
   COLUMNS_BY_SENSOR.append({'sensor':d['sensor'], 'right':d['right'].replace('Y','X'), 'left':d['left'].replace('Y','X')})
   COLUMNS_BY_SENSOR.append({'sensor':d['sensor'], 'right':d['right'].replace('Y','Z'), 'left':d['left'].replace('Y','Z')})   

DATA_DIR = os.path.join(r"C:\Users\sage\Documents\gate_analysis\raw_data")
GATE_CROSSING = -0.3
FREQUENCY = 128

