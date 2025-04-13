import pickle
import numpy as np
'''
with open('./experts/expert_transitions_2veh_98.pkl','rb') as file:
    data = pickle.load(file)
'''
with open('./experts/expert_transitions_2veh.pkl','rb') as file:
    data = pickle.load(file)

print(len(data))
print(data[1])
