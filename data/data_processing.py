import pandas as pd
import numpy as np
import copy
from Bio import SeqIO


def fa2dict(path):
    with open(path) as fa:
        fa_dict = {}
        for line in fa:
            # 去除末尾换行符
            line = line.replace("\n", "")
            if line.startswith(">"):
                # 去除 > 号
                seq_name = line[1:]
                fa_dict[seq_name] = ""
            else:
                # 去除末尾换行符并连接多行序列
                fa_dict[seq_name] += line.replace("\n", "")
    return fa_dict


database_df = pd.read_excel("RNADiseasev4.0_RNA-disease_experiment_piRNA.xlsx")
database_df.drop(database_df[pd.isna(database_df['DO ID'])].index, inplace=True)
all_p2seq_dup = {}
piRhsa_dict = {}
hsa_piR_dict = {}
hsapiR_dict = {}
piRhsa_base1 = {
    "RDID": [],
    "piRNA": [],
    "disease": [],
    "DOID": [],
    "MeSHID": [],
    "PMID": [],
    "score": [],
}
base2 = copy.deepcopy(piRhsa_base1)
dq_base3 = copy.deepcopy(piRhsa_base1)
hsa_piR_base4 = copy.deepcopy(piRhsa_base1)
hsapiR_base5 = copy.deepcopy(piRhsa_base1)
fr_base6 = copy.deepcopy(piRhsa_base1)
PIR_base7 = copy.deepcopy(piRhsa_base1)
other = copy.deepcopy(piRhsa_base1)


def update(dict, row):
    dict["RDID"].append(row["RDID"])
    dict["piRNA"].append(row["RNA Symbol"])
    dict["disease"].append(row["Disease Name"])
    dict["DOID"].append(row["DO ID"])
    dict["MeSHID"].append(row["MeSH ID"])
    dict["PMID"].append(row["PMID"])
    dict["score"].append(row["score"])


for index, row in database_df.iterrows():
    if row["specise"] != "Homo sapiens":
        continue
    if "." in row["RNA Symbol"]:
        continue
    if "piR-hsa-" in row["RNA Symbol"]:
        p_num = row["RNA Symbol"].split("-")[-1]
        new_num = str(int(p_num))
        row["RNA Symbol"] = row["RNA Symbol"].replace(p_num, new_num)
        update(piRhsa_base1, row)
    # elif "piRNA-" in row["RNA Symbol"]:
    #     row["RNA Symbol"] = row["RNA Symbol"].replace("piRNA-", "piR-")
    #     update(base2, row)
    # elif "pir-" in row["RNA Symbol"]:
    #     row["RNA Symbol"] = row["RNA Symbol"].replace("pir-", "piR-")
    #     update(base2, row)
    # elif "piR-" in row["RNA Symbol"]:
    #     update(base2, row)
    elif "DQ" in row["RNA Symbol"]:
        update(dq_base3, row)
    elif "hsa_piR_" in row["RNA Symbol"][:8]:
        update(hsa_piR_base4, row)
    elif "hsa-piR-" in row["RNA Symbol"]:
        if row["RNA Symbol"] == "hsa-piR-39888":
            continue
        update(hsapiR_base5, row)
    elif "FR" in row["RNA Symbol"]:
        update(fr_base6, row)
    elif "PIR" in row["RNA Symbol"]:
        update(PIR_base7, row)
    else:
        update(other, row)

piRhsa_uni = set(piRhsa_base1["piRNA"])
dq_uni = set(dq_base3["piRNA"])
hsa_piR_uni = set(hsa_piR_base4["piRNA"])
hsapiR_uni = set(hsapiR_base5["piRNA"])
fr_uni = set(fr_base6["piRNA"])
PIR_uni = set(PIR_base7["piRNA"])


sequence_fasta = r".\database\sequence.fasta"
xref_path = r".\database\xref.tsv"
sequence_tsv_path = r".\database\sequence.tsv"
hsa_v3_path = r".\database\hsa.v3.0.fa"
hsa_pirna_path = r".\database\hsa_pirna.fa"
piRNAdb_path = r".\database\piRNAdb.hsa.v1_7_6.fa"

fasta_sequences = SeqIO.parse(open(sequence_fasta), "fasta")
piRNAdb = SeqIO.parse(open(piRNAdb_path), "fasta")
xref_file = pd.read_csv(xref_path, sep="\t", header=0, index_col="xrefID")
sequence_tsv = pd.read_csv(sequence_tsv_path, sep="\t", header=0, index_col="id")

fr_dict = {}
dq_dict = {}


# 1
piRhsa_dict = fa2dict(hsa_v3_path)

for piRhsa_name in piRhsa_uni:
    all_p2seq_dup[piRhsa_name] = piRhsa_dict[piRhsa_name]

# 4
with open("./database/hsa_piR_to_piR_hsa.txt") as f:
    lines = f.readlines()
    for line in lines:
        line = line.replace("\n", "")
        hsa_piR_name, _, seq = line.split("\t")
        all_p2seq_dup[hsa_piR_name] = seq

# 5
hsapiR_dict = fa2dict(piRNAdb_path)
for hsapiR_name in hsapiR_uni:
    all_p2seq_dup[hsapiR_name] = hsapiR_dict[hsapiR_name]

# 3, 6
for fasta in fasta_sequences:
    name, sequence = fasta.description, str(fasta.seq)
    temp = name.split("|")
    type = temp[-1]
    if "piRNA" in type:
        fr_name = temp[0]
        dq_name = temp[1]
        fr_dict[fr_name] = sequence
        if "," in dq_name:
            dq_names = dq_name.split(",")
            for dq_name in dq_names:
                dq_dict[dq_name] = sequence
        else:
            dq_dict[dq_name] = sequence

for dq_name in dq_uni:
    all_p2seq_dup[dq_name] = dq_dict[dq_name]

for fr_name in fr_uni:
    all_p2seq_dup[fr_name] = fr_dict[fr_name]

# 7
for PIR_name in PIR_uni:
    xref_id = xref_file.loc[PIR_name]["id"]
    seq = sequence_tsv.loc[xref_id]["seq"]
    all_p2seq_dup[PIR_name] = seq

# func = lambda z: dict([(x, y) for y, x in z.items()])
# all_p2seq = func(func(all_p2seq_dup))
all_seq2p = {value: key for key, value in all_p2seq_dup.items()}
all_p2seq = {value: key for key, value in all_seq2p.items()}

piRhsa_base1_df = pd.DataFrame(piRhsa_base1)
dq_base3_df = pd.DataFrame(dq_base3)
hsa_piR_base4_df = pd.DataFrame(hsa_piR_base4)
hsapiR_base5_df = pd.DataFrame(hsapiR_base5)
fr_base6_df = pd.DataFrame(fr_base6)
PIR_base7_df = pd.DataFrame(PIR_base7)
comb_df = pd.concat(
    (
        piRhsa_base1_df,
        dq_base3_df,
        hsa_piR_base4_df,
        hsapiR_base5_df,
        fr_base6_df,
        PIR_base7_df,
    ),
    ignore_index=True,
)

comb_df=comb_df.replace('Early Hepatocellular Carcinoma', 'Hepatocellular Carcinoma')
all_seq = list(map(all_p2seq_dup.get, list(comb_df.piRNA), list(comb_df.piRNA)))
comb_df.piRNA = list(map(all_seq2p.get, all_seq, all_seq))
comb_df.to_csv("extracted_ass.csv", index=False)

uni_piRNA = list(set(comb_df.piRNA))
uni_disease = list(set(comb_df.disease))
d2doid = {}
adj_df_full = pd.DataFrame(index=uni_piRNA, columns=uni_disease, dtype=int)
for index, row in comb_df.iterrows():
    p_name = row["piRNA"]
    d_name = row["disease"]
    score = row["score"]
    adj_df_full.at[p_name, d_name] = 1
    d2doid[d_name] = row["DOID"]
    # adj.at[p_name, d_name] = score

adj_df_full[np.isnan(adj_df_full)] = 0
adj_df_full.sort_index(inplace=True)
adj_df_full.sort_index(inplace=True, axis=1)
adj_df_full.to_csv("adj_v4_full.csv")

del_columns = []
for column in adj_df_full.columns:
    if adj_df_full[column].sum() < 30:
        del_columns.append(column)
del_columns.remove('Lung Adenocarcinoma')
del_columns.remove('Lung Cancer')
del_columns.remove('Male Infertility')
adj_df = adj_df_full.drop(del_columns, axis="columns")
adj_df = adj_df[list(map(bool, list(adj_df.sum(1))))]
adj_df.to_csv("adj.csv")


all_p2seq_temp = {
    "piRNA": list(all_p2seq.keys()),
    "seq": list(all_p2seq.values()),
}
all_p2seq_df = pd.DataFrame(all_p2seq_temp)
all_p2seq_df.to_csv("all_piRNA_seq.csv", index=False)

all_d2doid_temp = {
    "piRNA": list(d2doid.keys()),
    "seq": list(d2doid.values()),
}
all_d2doid_df = pd.DataFrame(all_d2doid_temp)
all_d2doid_df.to_csv("all_doid.csv", index=False)

p2seq_temp = {
    "piRNA": list(adj_df.index),
    "seq": list(map(all_p2seq.get, list(adj_df.index), list(adj_df.index))),
}
p2seq_df = pd.DataFrame(p2seq_temp)
p2seq_df.to_csv("piRNA_seq.csv", index=False)

d2doid_temp = {
    "disease": list(adj_df.columns),
    "doid": list(map(d2doid.get, list(adj_df.columns), list(adj_df.columns))),
}
d2doid_df = pd.DataFrame(d2doid_temp)

d2doid_df.to_csv("doid.csv", index=False)

adj_df.index = list(map(all_p2seq.get, list(adj_df.index), list(adj_df.index)))
adj_df.columns = list(map(d2doid.get, list(adj_df.columns), list(adj_df.columns)))
adj_df.to_csv('databasev4_seq_doid.csv')

a=1