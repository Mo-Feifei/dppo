import numpy as np

data = np.load("/home/qxy/dppo/data/aliengo/trotting_straight/trotting_clean_50_scaled.npz")
states = data["states"]
action = data["actions"]

print(states[0,-39:])
print(action[0])
print(states[1,-39:])
print(action[1])
