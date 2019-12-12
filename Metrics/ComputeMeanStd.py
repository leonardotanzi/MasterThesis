import numpy as np


inc = [0.75,
0.74,
0.76,
0.74,
0.77
]

vgg = [0.84,
0.82,
0.80,
0.84,
0.79
]
res = [0.88,
0.85,
0.82,
0.84,
0.86
]


print(np.mean(inc))
print(np.std(inc))

print(np.mean(vgg))
print(np.std(vgg))

print(np.mean(res))
print(np.std(res))