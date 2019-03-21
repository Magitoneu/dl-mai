import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

path = '../datasets/Images'
classes = {}
all_classes = []
for root, dirs, files in os.walk(path, topdown=True):

    path_parts = root.split(os.sep)

    for file in files:
        if path_parts[-1] in classes:
            classes[path_parts[-1]] += 1
        else:
            classes[path_parts[-1]] = 1
        all_classes.append(path_parts[-1])

print(classes)
#plt.bar(classes.keys(), classes.values(), color='b')
#plt.xticks([i for i in range(0, 67)], classes.keys(), rotation='vertical')
#plt.show()
plt.figure(figsize=(20, 3))
ax = sns.countplot(all_classes)

ax.set_xticklabels(ax.get_xticklabels(), rotation=65)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
ax.set_title("Category Frequency vs Category")
print(list(classes.values()))
plt.axhline(np.mean(list(classes.values())), color='k', linewidth=2,)
plt.show()