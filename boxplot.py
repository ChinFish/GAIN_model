"""Plot impute preformance under different MAF.

Input file from: SNP_GAIN/accuracy.py
Input file format:
  - accuracy for MAF <= 0
  - accuracy for 0 < MAF <= 0.0001
  - ...
  - accuracy for 1 <= MAF
  - overall accuracy
  
  * No Header
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===
# Set MAF bin
# ===
maf_bin = ['[0]', '[0-0.0001]', '[0.0001-0.001]', '[0.001-0.01]', '[0.01-0.1]', '[0.1-1]', 'Overall Acc.']
maf_cnt = [0, 0, 0, 2, 116, 882]

# ===
# Read csv to pandas
# ===
main_df = pd.read_csv('/content/drive/MyDrive/GAIN/result/0503_twb_cnn/0503_chr22_test_TWB_accuracy_gain_3500.csv', sep=',', header=None)
main_df.columns = maf_bin
main_df

# ===
# Bar plot
# ===
main_df.boxplot()
plt.xticks(rotation=45)
plt.show()

# ===
# Plot MAF Bin
# ===
x = np.arange(len(maf_bin[:-1]))
plt.bar(x, maf_cnt)

# Show count
for i in x:
  plt.text(i, float(maf_cnt[i]), maf_cnt[i])

plt.xticks(x, maf_bin[:-1])
plt.xticks(rotation=45)
plt.show()