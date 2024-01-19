import pandas as pd
import numpy as np
import obonet
import networkx as nx
import math


url = "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo"
HDO_Net = obonet.read_obo(url)


def get_SV(disease, w):
    S = HDO_Net.subgraph(nx.descendants(HDO_Net, disease) | {disease})
    SV = dict()
    shortest_paths = nx.shortest_path(S, source=disease)
    for x in shortest_paths:
        SV[x] = math.pow(w, (len(shortest_paths[x]) - 1))
    return SV


def get_similarity(d1, d2, w):
    SV1 = get_SV(d1, w)
    SV2 = get_SV(d2, w)
    intersection_value = 0
    for disease in set(SV1.keys()) & set(SV2.keys()):
        intersection_value = intersection_value + SV1[disease]
        intersection_value = intersection_value + SV2[disease]
    return intersection_value / (sum(SV1.values()) + sum(SV2.values()))


def getDiSiNet(dilen, diseases, w):
    diSiNet = np.zeros((dilen, dilen))
    for d1 in range(dilen):
        if diseases[d1] in HDO_Net.nodes:
            for d2 in range(d1 + 1, dilen):
                if diseases[d2] in HDO_Net.nodes:
                    diSiNet[d1, d2] = diSiNet[d2, d1] = get_similarity(
                        diseases[d1], diseases[d2], w
                    )
    return diSiNet


doid_csv = pd.read_csv(r"doid.csv")

d_names = list(doid_csv.disease)
do_ids = list(doid_csv.doid)
s_d1 = getDiSiNet(len(do_ids), do_ids, 0.5)
np.fill_diagonal(s_d1, 1)
d2d = pd.DataFrame(s_d1, columns=d_names, index=d_names)
d2d.to_csv(r"d2d_do.csv")
