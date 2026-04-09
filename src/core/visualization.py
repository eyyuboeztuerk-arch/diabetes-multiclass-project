"""
visualization.py
Functions for data and results visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

CLASS_NAMES  = ['No Diabetes', 'Prediabetes', 'Diabetes']
CLASS_COLORS = ['#4575b4', '#fee090', '#d73027']  # Blue → Yellow → Red (ColorBrewer RdYlBu)




