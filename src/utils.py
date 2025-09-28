import torch
import random
from collections import defaultdict
# from torchmetrics.classification import MulticlassF1Score
from torch_geometric.utils import degree

def star_subgraph(adjacency_matrix, subgraph_size=4):
    num_nodes = adjacency_matrix.shape[0]
    subgraph_indices = []
    uncovered_neighbors = set(range(num_nodes))  # All nodes should be covered as neighbors at least once

    leaf_counts = defaultdict(int)

    seed_nodes = list(range(num_nodes))
    random.shuffle(seed_nodes)

    for center_node in seed_nodes:
        neighbors = [i for i in range(num_nodes) if adjacency_matrix[center_node, i] != 0 and i != center_node]
        k = subgraph_size - 1

        candidates = neighbors  # Already excludes center node

        # Case 1: Not enough neighbors → take all of them
        if len(candidates) <= k:
            sampled_neighbors = candidates

        else:
            available_new = list(set(candidates) & uncovered_neighbors)

            # Case 2a: enough new nodes → sample from them
            if len(available_new) >= k:
                sampled_neighbors = random.sample(available_new, k)

            # Case 2b: not enough new nodes → take all + fill from candidates
            else:
                sampled_neighbors = available_new
                remaining_k = k - len(sampled_neighbors)
                remaining_pool = list(set(candidates) - set(sampled_neighbors))
                remaining_pool.sort(key=lambda x: leaf_counts[x])

                sampled_neighbors += remaining_pool[:remaining_k]

        # Update uncovered neighbor set
        uncovered_neighbors -= set(sampled_neighbors)
        for node in sampled_neighbors:
            leaf_counts[node] += 1

        # Add center + its sampled neighbors
        subgraph = [center_node] + sampled_neighbors
        subgraph_indices.append(subgraph)

    return subgraph_indices


def train_graph(model, optimizer, loader, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_attr, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss)  * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test_graph(model, loader, criterion, device, num_classes=0):
    model.eval()
    total_loss = 0
    correct = 0
    
    all_preds = []
    all_labels = []
    f1 = 0
    
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_attr, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        total_loss += float(loss) * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        
        all_preds.append(pred)
        all_labels.append(data.y)
    # if num_classes:   
    #     all_preds = torch.cat(all_preds, dim=0)
    #     all_labels = torch.cat(all_labels, dim=0)
    #     f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    #     f1 = f1_metric(all_preds, all_labels)
    acc = correct / len(loader.dataset)
    return total_loss / len(loader.dataset), acc, f1

@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_attr, data.edge_index, data.batch)
        preds = out.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(data.y.cpu())
    return torch.cat(all_preds), torch.cat(all_labels)

def train_node(model, optimizer, data, criterion, device):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    out = model(data.x, data.edge_attr, data.edge_index, batch=None)  # batch is unused
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test_node(model, data, criterion, device, num_classes=0):
    model.eval()
    data = data.to(device)
    out = model(data.x, data.edge_attr, data.edge_index, batch=None)

    results = {}
    for split in ['train', 'val', 'test']:
        mask = getattr(data, f'{split}_mask')
        loss = criterion(out[mask], data.y[mask])
        pred = out[mask].argmax(dim=1)
        correct = (pred == data.y[mask]).sum().item()
        acc = correct / mask.sum().item()

        results[split] = {
            'loss': float(loss),
            'acc': acc,
        }

    return results


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, save_path="best_model.pt"):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)

class ToFloat:
    def __call__(self, data):
        data.x = data.x.float()             
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr.float()
        return data
    
class FixZINC(ToFloat):
    def __call__(self, data):
        data = super().__call__(data)  
        if data.y.dim() == 0:
            data.y = data.y.unsqueeze(-1)
        return data


class ConstantX(object):
    def __call__(self, data):
        if data.x is None:                            
            data.x = torch.ones((data.num_nodes, 1),
                                 dtype=torch.float)
        return data
    

class DegreeX(object):
    def __call__(self, data):
        if data.x is None:
            deg = degree(data.edge_index[0],
                         data.num_nodes,
                         dtype=torch.float)            
            data.x = deg.view(-1, 1)                  
        return data

class OneHotDegree(object):
    def __call__(self, data):
        if data.x is None:
            deg = degree(data.edge_index[0],
                         data.num_nodes).long()     
            num_classes = int(deg.max().item()) + 1
            data.x = torch.nn.functional.one_hot(deg,
                                                 num_classes)\
                                         .to(torch.float)
        return data