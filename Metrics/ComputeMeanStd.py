import numpy as np


inc = [0.65,
0.65,
0.66,
0.68,
0.62
]

vgg = [0.88,
0.87,
0.89,
0.86,
0.86
]

res = [0.58,
0.54,
0.62,
0.62,
0.68
]


print(np.mean(inc))
print(np.std(inc))

print(np.mean(vgg))
print(np.std(vgg))

print(np.mean(res))
print(np.std(res))