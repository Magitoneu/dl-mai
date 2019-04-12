import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

path = '../datasets/amazon_alexa.tsv'
classes = {}
all_classes = []

Tweet = pd.read_csv("../datasets/kindle_reviews.csv", delimiter=',')


for rating in Tweet['overall']:
    all_classes.append(rating)

    if rating in classes:
        classes[rating] += 1
    else:
        classes[rating] = 1
print(classes)
plt.figure(figsize=(15, 7))
ax = sns.countplot(all_classes)

ax.set_xticklabels(ax.get_xticklabels())
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_title("Category Frequency vs Category")
print(list(classes.values()))
plt.axhline(np.mean(list(classes.values())), color='k', linewidth=1,)
plt.show()