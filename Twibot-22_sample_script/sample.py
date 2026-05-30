import torch
import numpy as np
import json
import os
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from typing import Tuple

############################################
# 0. 配置
############################################

DATA_DIR = '../processed_data'
OUT_DIR  = './subset'

SEED_RATIO = 0.1
TRAIN_HOPS = 2
VAL_HOPS   = 2
TEST_HOPS  = 2

torch.manual_seed(42)
np.random.seed(42)

os.makedirs(OUT_DIR, exist_ok=True)

############################################
# 1. 加载图结构 & 标签
############################################

edge_index = torch.load(f'{DATA_DIR}/edge_index.pt').cpu()
edge_type  = torch.load(f'{DATA_DIR}/edge_type.pt').cpu()
labels     = torch.load(f'{DATA_DIR}/label.pt').cpu()

train_idx = torch.load(f'{DATA_DIR}/train_idx.pt')
val_idx   = torch.load(f'{DATA_DIR}/val_idx.pt')
test_idx  = torch.load(f'{DATA_DIR}/test_idx.pt')

############################################
# 2. 加载节点特征（多模态）
############################################

x_cat = torch.load(f'{DATA_DIR}/cat_properties_tensor.pt').cpu()
x_num = torch.load(f'{DATA_DIR}/num_properties_tensor.pt').cpu()
x_des = torch.load(f'{DATA_DIR}/des_tensor.pt').cpu()

# 原始推文（不进 Data）
with open(f'{DATA_DIR}/id_tweet.json', 'r', encoding='utf-8') as f:
    id_tweet = json.load(f)

############################################
# 3. 分层抽样
############################################

def stratified_sample(idx, labels, ratio):
    idx = idx.numpy()
    y = labels[idx].numpy()
    sampled = []

    for c in np.unique(y):
        c_idx = idx[y == c]
        k = max(1, int(len(c_idx) * ratio))
        sampled.append(
            np.random.choice(c_idx, k, replace=False)
        )

    return torch.tensor(np.concatenate(sampled), dtype=torch.long)

############################################
# 4. 限制边（防止跨 split）
############################################

def restrict_edges(edge_index, edge_type, allowed_nodes):
    mask = torch.isin(edge_index[0], allowed_nodes) & \
           torch.isin(edge_index[1], allowed_nodes)
    return edge_index[:, mask], edge_type[mask]

############################################
# 5. 构造一个 split 的子图（含多模态特征）
############################################

def build_split_data_transductive(seed_nodes, num_hops,
                                  edge_index, edge_type, labels,
                                  x_cat, x_num, x_des, id_tweet,
                                  train_idx, val_idx, test_idx,
                                  split_name):
    """
    Transductive setting:
    - 允许跨 split 连边
    - 通过 mask 控制训练 / 验证 / 测试
    """

    N = labels.size(0)

    # k-hop 子图（全局 ID）
    subset, sub_edge_index, _, edge_mask = k_hop_subgraph(
        seed_nodes,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=N
    )
    sub_edge_type = edge_type[edge_mask]

    # 🔹 关键修改：获取子图的节点数量
    num_nodes_subset = len(subset)

    # 构建 Data
    data = Data(
        edge_index=sub_edge_index,
        edge_type=sub_edge_type,
        y=labels[subset],
        num_nodes=num_nodes_subset
    )

    data.x_cat = x_cat[subset]
    data.x_num = x_num[subset]
    data.x_des = x_des[subset]

    # 🔹 构建 mask（这是关键）
    data.train_mask = torch.isin(subset, train_idx)
    data.val_mask   = torch.isin(subset, val_idx)
    data.test_mask  = torch.isin(subset, test_idx)

    # 🔹 切 tweets（subset 里包含 train/val/test 混合）
    sub_tweet = {str(int(uid)): id_tweet.get(str(int(uid)), []) for uid in subset.tolist()}
    with open(f'{OUT_DIR}/{split_name}_tweets.json', 'w', encoding='utf-8') as f:
        json.dump(sub_tweet, f, ensure_ascii=False, indent=2)

    # 保存 subset_idx
    torch.save(subset, f'{OUT_DIR}/{split_name}_subset_idx.pt')

    print(f"[INFO] {split_name}: nodes={data.num_nodes}, edges={data.num_edges}")
    print(f"       train/val/test = "
          f"{data.train_mask.sum().item()} / "
          f"{data.val_mask.sum().item()} / "
          f"{data.test_mask.sum().item()}")

    return data

############################################
# 6. 构造 Train / Val / Test
############################################

seed_train = stratified_sample(train_idx, labels, SEED_RATIO)
seed_val   = stratified_sample(val_idx,   labels, SEED_RATIO)
seed_test  = stratified_sample(test_idx,  labels, SEED_RATIO)

data = build_split_data_transductive(
    seed_nodes=torch.cat([seed_train, seed_val, seed_test]),
    num_hops=2,
    edge_index=edge_index,
    edge_type=edge_type,
    labels=labels,
    x_cat=x_cat,
    x_num=x_num,
    x_des=x_des,
    id_tweet=id_tweet,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
    split_name='full'
)


############################################
# 7. 保存 Data
############################################

torch.save(data, f'{OUT_DIR}/data_transductive.pt')

############################################
# 8. Sanity check
############################################

def summary(name, data):
    print(f'\n[{name}]')
    print(f'Nodes: {data.num_nodes}')
    print(f'Edges: {data.num_edges}')
    print(f'x_cat: {data.x_cat.shape}')
    print(f'x_num: {data.x_num.shape}')
    print(f'x_des: {data.x_des.shape}')
    print(f'Label dist: {torch.bincount(data.y)}')

summary('All', data)


print('\nAll done.')
