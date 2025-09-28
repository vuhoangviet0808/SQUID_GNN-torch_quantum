import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import MLP, global_add_pool, global_mean_pool, global_max_pool   

from utils import star_subgraph


def message_passing_pqc(strong, twodesign, inits, wires):
    edge, center, neighbor, ancilla1, ancilla2 = wires
    ##
    # qml.StronglyEntanglingLayers(weights=strong[0], wires=[edge, center, neighbor])
    ##
    # qml.CRX(phi=inits[0,0],wires=[center, edge])
    # qml.CRX(phi=inits[0,1],wires=[center, neighbor])
    # qml.StronglyEntanglingLayers(weights=strong[0], wires=[edge, ancilla])
    # qml.StronglyEntanglingLayers(weights=strong[0], wires=[neighbor, ancilla])
    # qml.StronglyEntanglingLayers(weights=strong[0], wires=[center, ancilla])
    # qml.StronglyEntanglingLayers(weights=strong[1], wires=[edge, center])
    # qml.StronglyEntanglingLayers(weights=strong[2], wires=[center, neighbor])
    
    ## only rotation
    qml.CRX(inits[0, 0], wires=[neighbor, ancilla1])
    qml.CRY(inits[0, 1], wires=[edge, ancilla1])
    qml.CRZ(inits[0, 2], wires=[neighbor, ancilla2])
    qml.CRY(inits[0, 3], wires=[edge, ancilla2])
    qml.StronglyEntanglingLayers(weights=strong[0], wires=[edge, neighbor, ancilla1])
    # qml.adjoint(qml.StronglyEntanglingLayers(weights=strong[0], wires=[edge, neighbor, ancilla1]))
    qml.StronglyEntanglingLayers(weights=strong[1], wires=[ancilla1, neighbor, ancilla2])
    # qml.adjoint(qml.StronglyEntanglingLayers(weights=strong[1], wires=[ancilla1, neighbor, ancilla2]))


def qgcn_enhance_layer(inputs, spreadlayer, strong, twodesign, inits, update):
    edge_feat_dim = feat_dim = node_feat_dim = 2
    inputs = inputs.reshape(-1,feat_dim)
    
    # The number of avaible nodes and edges
    total_shape = inputs.shape[0]
    num_nodes = (total_shape+1)//2
    num_edges = num_nodes - 1
    
    adjacency_matrix, vertex_features = inputs[:num_edges,:], inputs[num_edges:,:]

    # The number of qubits assiged to each node and edge
    num_qbit = spreadlayer.shape[1]
    num_nodes_qbit = (num_qbit+1)//2
    num_edges_qbit = num_nodes_qbit - 1
    
    center_wire = num_edges_qbit
    
    
    for i in range(num_edges):
        qml.RY(adjacency_matrix[i][0], wires=i)
        qml.RZ(adjacency_matrix[i][1], wires=i)
        # qml.RX(adjacency_matrix[i][2], wires=i)
    
    for i in range(num_nodes):
        qml.RY(vertex_features[i][0], wires=center_wire+i)
        qml.RZ(vertex_features[i][1], wires=center_wire+i)
        # qml.RX(vertex_features[i][2], wires=center_wire+i)
    
    
    for i in range(num_edges):

        message_passing_pqc(strong=strong, twodesign=twodesign, inits=inits, 
                            wires=[i, center_wire, center_wire+i+1, num_qbit, num_qbit+1])

    qml.StronglyEntanglingLayers(
        weights=update[0], 
        wires=[center_wire, num_qbit, num_qbit+1]
        )
    # probs = qml.probs(wires=[center_wire, num_qbit, num_qbit+1])
    # return probs
    # expval = [qml.expval(qml.PauliZ(w)) for w in [center_wire, num_qbit, num_qbit+1]]
    expval = [
        qml.expval(qml.PauliX(center_wire)),
        # qml.expval(qml.PauliY(center_wire)),
        # qml.expval(qml.PauliZ(center_wire)),
        # qml.expval(qml.PauliX(num_qbit)),
        # qml.expval(qml.PauliY(num_qbit)),
        # qml.expval(qml.PauliZ(num_qbit)),
        # qml.expval(qml.PauliX(num_qbit+1)),
        # qml.expval(qml.PauliY(num_qbit+1)),
        # qml.expval(qml.PauliZ(num_qbit+1)),
    ]
    return expval


def small_normal_init(tensor):
    return torch.nn.init.normal_(tensor, mean=0.0, std=0.1)

def uniform_pi_init(tensor):
    return nn.init.uniform_(tensor, a=0.0, b=np.pi)

def identity_block_init(tensor):
    with torch.no_grad():
        tensor.zero_()
        if tensor.ndim < 1:
            return tensor  # scalar param

        # Total number of parameters
        total_params = tensor.numel()
        num_active = max(1, total_params // 3)

        # Flatten, randomize, and reshape
        flat = tensor.view(-1)
        active_idx = torch.randperm(flat.shape[0])[:num_active]
        flat[active_idx] = torch.randn_like(flat[active_idx]) * 0.1

        return tensor
    
def input_process(tensor):
    # return torch.clamp(tensor, -1.0, 1.0) * np.pi
    return torch.tanh(tensor) * np.pi


class QGNNGraphClassifier(nn.Module):
    def __init__(self, q_dev, w_shapes, node_input_dim=1, edge_input_dim=1,
                 graphlet_size=4, hop_neighbor=1, num_classes=2, one_hot=0):
        super().__init__()
        self.hidden_dim = 128
        self.graphlet_size = graphlet_size
        self.one_hot = one_hot
        self.hop_neighbor = hop_neighbor
        self.pqc_dim = 2 # number of feat per pqc for each node
        self.chunk = 1
        self.final_dim = self.pqc_dim * self.chunk # 2
        self.pqc_out = 1 # probs?
        
        
        self.qconvs = nn.ModuleDict()
        self.upds = nn.ModuleDict()
        self.aggs = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        
        if self.one_hot:
            self.node_input_dim = 1
            self.edge_input_dim = 1
        else:
            self.node_input_dim = node_input_dim
            self.edge_input_dim = edge_input_dim if edge_input_dim > 0 else 1

        
        self.input_node = MLP(
                    [self.node_input_dim, self.hidden_dim, self.final_dim],
                    act='leaky_relu', 
                    norm=None, dropout=0.3
            )

        self.input_edge = MLP(
                    [self.edge_input_dim, self.hidden_dim, self.pqc_dim],
                    act='leaky_relu', 
                    norm=None, dropout=0.3
            )
        
        for i in range(self.hop_neighbor):
            qnode = qml.QNode(qgcn_enhance_layer, q_dev,  interface="torch")
            self.qconvs[f"lay{i+1}"] = qml.qnn.TorchLayer(qnode, w_shapes, uniform_pi_init)
            
            self.upds[f"lay{i+1}"] = MLP(
                    [self.pqc_dim + self.pqc_out, self.hidden_dim, self.pqc_dim],
                    act='leaky_relu', 
                    norm=None, dropout=0.3
            )
            
            self.norms[f"lay{i+1}"] = nn.LayerNorm(self.pqc_dim)
            
        self.graph_head = MLP(
                [self.final_dim, num_classes, num_classes],
                act='leaky_relu', 
                norm=None, dropout=0.3
        ) 
        
    def forward(self, node_feat, edge_attr, edge_index, batch):
        edge_index = edge_index.t()
        num_nodes = node_feat.size(0)
        num_nodes_model = self.graphlet_size
        num_edges_model = self.graphlet_size - 1
        
        if edge_attr is None:
            edge_attr = torch.ones((edge_index.size(0), self.edge_input_dim), device=node_feat.device)
        
        edge_features = edge_attr.float()
        node_features = node_feat.float()
        
        edge_features = self.input_edge(edge_features)
        node_features = self.input_node(node_features)
        
        node_features = input_process(node_features)
        # # node_features = node_features + 0.01 * torch.randn_like(node_features)
        edge_features = input_process(edge_features)
        
        
        idx_dict = {
            (int(u), int(v)): i
            for i, (u, v) in enumerate(edge_index.tolist())
        }
        

        adj_mtx = torch.zeros((num_nodes, num_nodes), dtype=torch.int)
        adj_mtx[edge_index[:, 0], edge_index[:, 1]] = 1
        adj_mtx[edge_index[:, 1], edge_index[:, 0]] = 1
        
        
        for i in range(self.hop_neighbor):
            subgraphs = star_subgraph(adj_mtx.cpu().numpy(), subgraph_size=self.graphlet_size)
            node_upd = torch.zeros((num_nodes, self.final_dim), device=node_features.device)
            q_layer = self.qconvs[f"lay{i+1}"]
            upd_layer = self.upds[f"lay{i+1}"]
            norm_layer = self.norms[f"lay{i+1}"]

            # updates_node = node_features.clone() 
            
            centers = []
            updates = []
            
            for sub in subgraphs:
                center, *neighbors = sub

                n_feat = node_features[sub] 
                # edge_idxs = [ idx_dict[(center, int(n))] for n in neighbors ]
                edge_idxs = [
                    idx_dict[(min(center, int(n)), max(center, int(n)))] 
                    for n in neighbors 
                ]
                e_feat    = edge_features[edge_idxs]  
                inputs = torch.cat([e_feat, n_feat], dim=0)        

                all_msg = q_layer(inputs.flatten())
                aggr = all_msg
                update_vec = upd_layer(torch.cat([node_features[center], aggr], dim=0))
            
                centers.append(center)
                updates.append(update_vec)
            
            centers = torch.tensor(centers, device=node_features.device)
            updates = torch.stack(updates, dim=0) 
            updates_node = torch.zeros_like(node_features)
            updates_node = updates_node.index_add(0, centers, updates)
            
            # node_features = norm_layer(updates_node + node_features)    
            node_features = updates_node + node_features
        graph_embedding = global_mean_pool(node_features, batch)
        
        return self.graph_head(graph_embedding)