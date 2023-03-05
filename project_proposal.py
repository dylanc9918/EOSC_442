import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
import glob
import os

path = r'data'
all_files = glob.glob(os.path.join(path, "/*.csv"))

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)


df = pd.read_csv()
