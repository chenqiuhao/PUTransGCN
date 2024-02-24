import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("result_compare195.xlsx", sheet_name="all_result", index_col=0)
result = df[["rank_idx", "auc", "aupr"]]
auc_list_str = result.loc[[f"PUTransGCN_comb_{i}" for i in range(1, 11)]][
    "auc"
].tolist()
auc_list = [float(x.split("±")[0]) for x in auc_list_str]
plt.plot(range(1, 11), auc_list, marker=".")
plt.xticks(range(1, 11), [f"{i}%" for i in range(1, 11)])
plt.xlabel("spy percentage")
plt.ylabel("AUC value")
plt.grid(axis="y")
plt.savefig("spy_percent")

plt.show()
a = 1
