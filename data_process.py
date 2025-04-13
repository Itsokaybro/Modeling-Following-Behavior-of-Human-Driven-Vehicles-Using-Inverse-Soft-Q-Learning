import pandas as pd
import numpy as np
import pickle


with open('./data_origin/hh_traj_total_5.pkl', 'rb') as file:
    data = pickle.load(file) #spacing v_rel ...

#print(data)

#v_rel = []
spacing = []
lv_v = []
fv_v = []
action = []
count = 0

for i in data:
    for j in i:
        state, acc = j
        spacing.append(state[0])
        lv_v.append(state[1])
        fv_v.append(state[2])
        action.append(acc)
    
#print(len(v_rel))

df = pd.DataFrame(columns=['Vehicle_ID', 'spacing', 'lv_v', 'fv_v', 'action'])
ID = 0
for k in range(len(spacing)):
    if k % 99 == 0:
        ID += 1
        if ID == 100: break
    new_row = [ID, spacing[k], lv_v[k], fv_v[k], action[k]]
    df.loc[len(df)] = new_row

csv_filename = 'hh_data_2.csv'
df.to_csv(csv_filename, index=False) 
