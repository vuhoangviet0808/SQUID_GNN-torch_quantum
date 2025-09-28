import torch
import os
from torch_geometric.datasets import TUDataset, ZINC, Planetoid, WebKB
from torch_geometric.loader import DataLoader
from utils import FixZINC, ConstantX, DegreeX

def load_dataset(name, path='../data', train_size=None, test_size=None, batch_size=32):
    name = name.upper()
    task_type = 'graph'

    if name in ['MUTAG', 'ENZYMES', 'PROTEINS']:
        dataset = TUDataset(os.path.join(path, 'TUDataset'), name=name)
        torch.manual_seed(1712)
        dataset = dataset.shuffle()
    elif name == 'REDDIT-BINARY':
        dataset = TUDataset(os.path.join(path, 'TUDataset'), name=name, transform=ConstantX())
        torch.manual_seed(1712)
        dataset = dataset.shuffle()

    elif name == 'ZINC':
        dataset = ZINC(root=os.path.join(path, 'ZINC'), transform=FixZINC())
        torch.manual_seed(1712)
        dataset = dataset.shuffle()

    elif name in ['CORA', 'CITESEER', 'PUBMED']:
        dataset = Planetoid(root=os.path.join(path, 'Planetoid'), name=name)
        data = dataset[0]
        return dataset, data, data, 'node'

    elif name in ['CORNELL', 'WISCONSIN']:
        dataset = WebKB(root=os.path.join(path, 'WebKB'), name=name.lower(), geom_gcn_preprocess=True)
        data = dataset[0]
        return dataset, data, data, 'node'

    else:
        raise ValueError(f"Dataset '{name}' not supported.")

    if train_size and test_size:
        train_dataset = dataset[:train_size]
        test_dataset = dataset[train_size:train_size + test_size]
    else:
        train_dataset = dataset[:int(0.8 * len(dataset))]
        test_dataset = dataset[int(0.8 * len(dataset)):]

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return dataset, train_loader, test_loader, task_type


def eval_dataset(name, path='../data', eval_size=None, batch_size=32, seed=1309):
    name = name.upper()
    task_type = 'graph'

    if name in ['MUTAG', 'ENZYMES', 'PROTEINS']:
        dataset = TUDataset(os.path.join(path, 'TUDataset'), name=name)
        torch.manual_seed(seed)
        dataset = dataset.shuffle()
        eval_set = dataset[:eval_size] if eval_size else dataset[int(0.8 * len(dataset)):]
        eval_loader = DataLoader(eval_set, batch_size=batch_size)

    elif name == 'REDDIT-BINARY':
        dataset = TUDataset(os.path.join(path, 'TUDataset'), name=name, transform=ConstantX())
        torch.manual_seed(1712)
        dataset = dataset.shuffle()
        eval_set = dataset[:eval_size] if eval_size else dataset[int(0.8 * len(dataset)):]
        eval_loader = DataLoader(eval_set, batch_size=batch_size)
    elif name == 'ZINC':
        dataset = ZINC(root=os.path.join(path, 'ZINC'))
        torch.manual_seed(seed)
        dataset = dataset.shuffle()
        eval_set = dataset[:eval_size] if eval_size else dataset[int(0.8 * len(dataset)):]
        eval_loader = DataLoader(eval_set, batch_size=batch_size)

    elif name in ['CORA', 'CITESEER', 'PUBMED']:
        dataset = Planetoid(root=os.path.join(path, 'Planetoid'), name=name)
        eval_loader = dataset[0]
        task_type = 'node'

    elif name in ['CORNELL', 'WISCONSIN']:
        dataset = WebKB(root=os.path.join(path, 'WebKB'), name=name.lower(), geom_gcn_preprocess=True)
        eval_loader = dataset[0]
        task_type = 'node'

    else:
        raise ValueError(f"Dataset '{name}' not supported.")

    return eval_loader

def random_split(data, train_ratio=0.6, val_ratio=0.2, seed=42):
    torch.manual_seed(seed)
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)

    train_end = int(train_ratio * num_nodes)
    val_end = train_end + int(val_ratio * num_nodes)

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[perm[:train_end]] = True
    data.val_mask[perm[train_end:val_end]] = True
    data.test_mask[perm[val_end:]] = True
    return data


# def load_dataset(name, path='../data', train_size=None, eval_size=None, test_size=None, batch_size=32):
#     name = name.upper()

#     if name in ['MUTAG', 'ENZYMES', 'PROTEINS']:
#         dataset = TUDataset(os.path.join(path, 'TUDataset'), name=name)
#         torch.manual_seed(1712)
#         dataset = dataset.shuffle()

#         total_len = len(dataset)
#         if train_size and eval_size and test_size:
#             train_dataset = dataset[:train_size]
#             eval_dataset = dataset[train_size:train_size + eval_size]
#             test_dataset = dataset[train_size + eval_size:train_size + eval_size + test_size]
#         else:
#             train_end = int(0.7 * total_len)
#             eval_end = int(0.85 * total_len)
#             train_dataset = dataset[:train_end]
#             eval_dataset = dataset[train_end:eval_end]
#             test_dataset = dataset[eval_end:]

#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#         task_type = 'graph'

#     elif name in ['CORA', 'CITESEER', 'PUBMED']:
#         dataset = Planetoid(root=os.path.join(path, 'Planetoid'), name=name)
#         data = dataset[0]  # only one graph
#         train_loader = eval_loader = test_loader = data
#         task_type = 'node'
#     else:
#         raise ValueError(f"Dataset '{name}' not supported.")

#     return dataset, train_loader, eval_loader, test_loader, task_type

