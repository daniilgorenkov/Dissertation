from PDS import PDS as pds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

result1 = pds.get_result("empty", "straight", "normal", 10, "gost", "vertical")
result2 = pds.get_result("empty", "straight", "normal", 80, "gost", "vertical")

speeds = [i for i in range(10, 130, 10)]

valid_result = result1[result1.index > 20]

n = len(valid_result.index)
maximum = valid_result.max()
minimum = valid_result.min()

generated = np.random.uniform(low=minimum, high=maximum, size=(1, n))

valid_result.shape, generated.shape

valid_result

valid_result["generated"] = generated.reshape(-1, 1)

# plt.figure().set_size_inches(12,8)
plt.subplot(1, 2, 1)
sns.lineplot(valid_result["empty_straight_normal_10_gost"])


plt.subplot(1, 2, 2)
sns.lineplot(valid_result["generated"])
plt.show()
