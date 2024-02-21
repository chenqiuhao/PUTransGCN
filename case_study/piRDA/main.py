import numpy as np
from sklearn import tree, svm
from utils import *
from model import *
from sklearn.model_selection import StratifiedKFold
import warnings
import pickle

warnings.filterwarnings("ignore")
seed_everything(42)
device = torch.device("cuda")
import os

path = "scores/"
if not os.path.exists(path):
    os.makedirs(path)

piRNA_seq_csv = pd.read_csv(r"..\data\piRNA_seq.csv")
piRNA_seq_dict = dict(piRNA_seq_csv.values)
adj_df = pd.read_csv(r"..\data\adj.csv", index_col=0)
adj = adj_df.values
num_p, num_d = adj.shape

with open(r"../data/p_gip.pickle", "rb") as f:
    p_gip = pickle.load(f)
with open(r"../data/d_gip.pickle", "rb") as f:
    d_gip = pickle.load(f)
pos_ij = np.argwhere(adj == 1)
unlabelled_ij = np.argwhere(adj == 0)

epoch_num = 200
logger = Logger(5)

char_map = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}

p_onehot2d_np = np.zeros((num_p, 32, 4))
p_onehot_np = np.zeros((num_p, 32 * 4))

j = 0
for name, seq in piRNA_seq_dict.items():
    seq = seq.ljust(32, "N")
    temp = np.array(list(map(lambda x: char_map[x], list(seq))))
    p_onehot2d_np[j] = temp
    emb = np.array(list(itertools.chain(*temp)))
    p_onehot_np[j] = emb
    j = j + 1

d_onehot_np = np.diag([1] * num_d)

p_onehot2d = torch.FloatTensor(p_onehot2d_np).to(device)
d_onehot = torch.FloatTensor(d_onehot_np).to(device)

ass_feat_array = np.zeros((num_p, num_d, (32 * 4 + num_d)))
for i in range(num_p):
    for j in range(num_d):
        ass_feat_array[i, j] = np.append(d_onehot_np[j], p_onehot_np[i])


bootstrap_idx = np.random.choice(
    np.arange(len(unlabelled_ij)), replace=False, size=len(pos_ij)
)
unlabelled_feat = ass_feat_array[tuple(list(unlabelled_ij.T))]
pos_feat = ass_feat_array[tuple(list(pos_ij.T))]
bootstrap_ij = unlabelled_ij[bootstrap_idx]
bootstrap_feat = ass_feat_array[tuple(list(bootstrap_ij.T))]

svm_train = np.concatenate((pos_feat, bootstrap_feat), axis=0)
svm_label = [1] * len(pos_feat) + [0] * len(bootstrap_feat)

regressor = svm.SVC(probability=True)

regressor.fit(svm_train, svm_label)

score = regressor.predict_proba(unlabelled_feat)[:, 1]

# score = np.arange(len(unlabelled_train_ij))
sorted_nums = sorted(enumerate(score), key=lambda x: x[1])
idx = [i[0] for i in sorted_nums]

rn_idx = idx[int(len(idx) / 3) : int(len(idx) / 3 * 2)]
rn_ij = unlabelled_ij[rn_idx]

rn_ij_4 = np.array_split(rn_ij, 4)

for j in range(1):
    print(j)
    rn_ij_1 = rn_ij_4[j]

    model = PiRDA(p_onehot2d, d_onehot)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(piRDA.parameters(), lr=lr)
    dense_layers = nn.ModuleList([model.layer1, model.layer2])
    special_layers_params = list(map(id, dense_layers.parameters()))
    base_params = filter(
        lambda p: id(p) not in special_layers_params, model.parameters()
    )

    optimizer = torch.optim.Adam(
        [
            {"params": base_params},
            {"params": dense_layers.parameters(), "weight_decay": 1e-6},
        ],
        lr=0.001,
    )

    train_ij_np = np.concatenate((pos_ij, rn_ij_1), axis=0)
    train_label_np = adj[tuple(list(train_ij_np.T))]
    # test_ij_np = np.concatenate((unlabelled_ij), axis=0)
    test_ij_np = unlabelled_ij
    test_label_np = adj[tuple(list(test_ij_np.T))]

    train_ij = torch.IntTensor(train_ij_np).to(device)
    test_ij = torch.IntTensor(test_ij_np).to(device)
    train_label = torch.FloatTensor(train_label_np).to(device)
    test_label = torch.FloatTensor(test_label_np).to(device)

    for epoch in range(epoch_num):
        model.train()
        optimizer.zero_grad()
        train_pred = model(train_ij)
        train_loss = loss_fn(train_pred, train_label)
        train_loss.backward()
        optimizer.step()
        # model.eval()
        # test_pred = model(test_ij)
        # test_loss = loss_fn(test_pred, test_label)


model.eval()
test_pred = model(test_ij)
scores = test_pred.cpu().detach().numpy()
pred_np = np.zeros_like(adj)
pred_np[tuple(list(pos_ij.T))] = 1
pred_np[tuple(list(test_ij_np.T))] = scores
np.savetxt('pred_np.csv', pred_np, delimiter=",")
