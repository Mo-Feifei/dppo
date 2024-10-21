import numpy as np

data = np.load("/home/qxy/dppo/data/aliengo/trotting_straight/trotting_clean_50_scaled.npz")
states = data["states"]
action = data["actions"]

print(states[0,-78:])