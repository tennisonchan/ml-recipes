# https://www.youtube.com/watch?v=N9fDIAflCMY

import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

greyhound_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 28 + 4 * np.random.randn(labs)

plt.hist([greyhound_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()

# Features capture different types of information
# What types of features - hair length, speed, weight
# Avoid useless features - it can hurt the accuracy as purely useful by accident
# Independent features are the best
# Avoid redundant features - might double count the weight of the feature
# Features which are easy to understand - distance vs. geo location

# # Ideal features
# - Informative
# - Independent
# - Simple