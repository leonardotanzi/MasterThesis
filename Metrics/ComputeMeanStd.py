import numpy as np


inc = [0.854, 0.826, 0.809, 0.847, 0.843]
vgg = [0.861, 0.822, 0.784, 0.850, 0.826]
res = [0.815, 0.822, 0.857, 0.798, 0.802]

precisionA = [83.18, 83.17, 87.10, 86.14, 78.95]
sensitivityA = [89, 84, 81, 87, 90]
specifityA = [90.32, 90.61, 93.10, 86.57, 86.96]


precisionB = [90.12, 84.34, 91.78, 89.29, 91.78]
sensitivityB = [73, 70, 67, 75, 67]
specifityB = [95.83, 93.19, 96.70, 95.21, 96.83]


precisionU = [84.82, 81.03, 70.9, 80, 82.30]
sensitivityU = [95, 94,  95, 92, 93]
specifityU = [90.50, 87.50, 79.14, 87.57, 88.70]

print(np.mean(precisionU))
print(np.std(precisionU))

print(np.mean(sensitivityU))
print(np.std(sensitivityU))

print(np.mean(specifityU))
print(np.std(specifityU))