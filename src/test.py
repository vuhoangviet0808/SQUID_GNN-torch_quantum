import torch
import torch.nn as nn
import numpy as numpy
# import pennylane as qml
# from pennylane import numpy as np
# from torch_scatter import scatter_add
import torch.nn.functional as F
from torch_geometric.nn import MLP, global_add_pool, global_mean_pool, global_max_pool   

from utils import star_subgraph


class HandcraftGNN(nn.Module):
    def __init__(self, q_dev, w_shapes, node_input_dim=1, edge_input_dim=1,
                 graphlet_size=4, hop_neighbor=1, num_classes=2, one_hot=0):
        super().__init__()
        self.pqc_dim = 3 # number of feat for each node
        self.hidden_dim = 128
        self.graphlet_size = graphlet_size
        self.one_hot = one_hot
        self.hop_neighbor = hop_neighbor
        
        self.msg_dim = 2
        
        self.upds = nn.ModuleDict()
        self.msgs = nn.ModuleDict()
        self.aggs = nn.ModuleDict()
        

        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim if edge_input_dim > 0 else 1

        
        self.input_node = MLP(
                    [self.node_input_dim, self.hidden_dim, self.pqc_dim],
                    act=nn.LeakyReLU(0.1),
                    norm=None, dropout=0.2
            )
        
        self.input_edge = MLP(
                    [self.edge_input_dim, self.hidden_dim, self.pqc_dim],
                    act=nn.LeakyReLU(0.1),
                    norm=None, dropout=0.2
            )
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.pqc_dim) for _ in range(self.hop_neighbor)
        ])
        
        for i in range(self.hop_neighbor):
            
            self.upds[f"lay{i+1}"] = MLP(
                    [self.pqc_dim + self.msg_dim, self.hidden_dim, self.pqc_dim],
                    act=nn.LeakyReLU(0.1),
                    norm=None, dropout=0.2
            )
            
            
            self.msgs[f"lay{i+1}"] = MLP(
                    [self.pqc_dim*2, self.hidden_dim, self.msg_dim],
                    act=nn.LeakyReLU(0.1),
                    norm=None, dropout=0.2
            )


        self.graph_head = MLP(
                [self.pqc_dim, self.hidden_dim, num_classes],
                act=nn.LeakyReLU(0.1), 
                norm=None, dropout=0.2
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
        
        idx_dict = {
            (int(u), int(v)): i
            for i, (u, v) in enumerate(edge_index.tolist())
        }
            
        
        # edge_attributes = torch.zeros(num_nodes, num_nodes, self.final_dim, device=edge_attr.device)
        # edge_attributes[edge_index[:, 0], edge_index[:, 1]] = edge_features
        # edge_attributes[edge_index[:, 1], edge_index[:, 0]] = edge_features

        edge_np = edge_index.cpu().numpy() 
        rows = edge_np[:, 0]               
        cols = edge_np[:, 1]               

        adj_mtx = numpy.zeros((num_nodes, num_nodes), dtype=int)
        adj_mtx[rows, cols] = 1
        adj_mtx[cols, rows] = 1
        
        subgraphs = star_subgraph(adj_mtx, subgraph_size=self.graphlet_size)
        
        for i in range(self.hop_neighbor):
            upd_layer = self.upds[f"lay{i+1}"]
            msg_layer = self.msgs[f"lay{i+1}"]
            
            norm_layer = self.norms[i]
            
            # updates = [[] for _ in range(num_nodes)]  # each list holds candidate updates
            updates_node = torch.zeros_like(node_features) ## UPDATES
            
            centers = []
            updates = []
            for sub in subgraphs:
                center = sub[0]
                neighbors = sub[1:]

                n_feat = node_features[neighbors] 
                # e_feat = edge_attributes[center, neighbors] 
                edge_idxs = [ idx_dict[(center, int(n))] for n in neighbors ]
                e_feat    = edge_features[edge_idxs]  
                
                inputs = torch.cat([e_feat, n_feat], dim=1)      
                all_msg = msg_layer(inputs)   
                aggr = torch.sum(all_msg, dim=0)   
                new_center  = upd_layer(torch.cat([node_features[center], aggr], dim=0))
                
                centers.append(center)
                updates.append(new_center)
                
                # updates[center].append(new_center)
                # updates_node[center] = updates_node[center] + new_center  
            centers = torch.tensor(centers, device=node_features.device)
            updates = torch.stack(updates, dim=0) 
            
            # updates_node = scatter_add(updates, centers, dim=0,
            #            dim_size=node_features.size(0))
            
            updates_node = torch.zeros_like(node_features)
            updates_node = updates_node.index_add(0, centers, updates)
                        
            node_features = updates_node + node_features # norm_layer(updates_node + node_features)

        graph_embedding = global_add_pool(node_features, batch)
        
        return self.graph_head(graph_embedding)


class HandcraftGNN_NodeClassification(nn.Module):
    def __init__(self, q_dev, w_shapes, node_input_dim=1, edge_input_dim=1,
                 graphlet_size=4, hop_neighbor=1, num_classes=2, one_hot=0):
        super().__init__()
        self.pqc_dim = 1 # number of qubits for each node
        self.hidden_dim = 64
        self.graphlet_size = graphlet_size
        self.one_hot = one_hot
        self.hop_neighbor = hop_neighbor
        self.final_dim = 2**self.pqc_dim
        
        self.upds = nn.ModuleDict()
        self.msgs = nn.ModuleDict()
        self.aggs = nn.ModuleDict()
        

        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim if edge_input_dim > 0 else 1

        
        self.input_node = nn.Linear(in_features=self.node_input_dim, out_features=self.final_dim)
        self.input_edge = nn.Linear(in_features=self.edge_input_dim, out_features=self.final_dim)
        
        for i in range(self.hop_neighbor):
            
            self.upds[f"lay{i+1}"] = MLP(
                    [self.final_dim*2, self.hidden_dim, self.final_dim],
                    act=nn.LeakyReLU(0.1),
                    norm=None, dropout=0.2
            )
            
            self.msgs[f"lay{i+1}"] = MLP(
                    [self.final_dim*2, self.hidden_dim, self.final_dim],
                    act=nn.LeakyReLU(0.1),
                    norm=None, dropout=0.2
            )

        self.final = MLP(
                [self.final_dim, self.hidden_dim, num_classes],
                act=nn.LeakyReLU(0.1), 
                norm=None, dropout=0.2
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
        
        
        idx_dict = {
            (int(u), int(v)): i
            for i, (u, v) in enumerate(edge_index.tolist())
        }
            
        
        # edge_attributes = torch.zeros(num_nodes, num_nodes, self.final_dim, device=edge_attr.device)
        # edge_attributes[edge_index[:, 0], edge_index[:, 1]] = edge_features
        # edge_attributes[edge_index[:, 1], edge_index[:, 0]] = edge_features

        edge_np = edge_index.cpu().numpy() 
        rows = edge_np[:, 0]               
        cols = edge_np[:, 1]               

        adj_mtx = numpy.zeros((num_nodes, num_nodes), dtype=int)
        adj_mtx[rows, cols] = 1
        adj_mtx[cols, rows] = 1
        
        subgraphs = star_subgraph(adj_mtx, subgraph_size=self.graphlet_size)
        
        for i in range(self.hop_neighbor):
            upd_layer = self.upds[f"lay{i+1}"]
            msg_layer = self.msgs[f"lay{i+1}"]
            
            # updates = [[] for _ in range(num_nodes)]  # each list holds candidate updates
            updates_node = node_features.clone() ## UPDATES
            for sub in subgraphs:
                center = sub[0]
                neighbors = sub[1:]

                n_feat = node_features[neighbors] 
                # e_feat = edge_attributes[center, neighbors] 
                edge_idxs = [ idx_dict[(center, int(n))] for n in neighbors ]
                e_feat    = edge_features[edge_idxs]  
                
                inputs = torch.cat([e_feat, n_feat], dim=1)      
                all_msg = msg_layer(inputs)   
                aggr = torch.sum(all_msg, dim=0)   
                
                new_center  = upd_layer(torch.cat([node_features[center], aggr], dim=0))
                # updates[center].append(new_center)
                updates_node[center] = updates_node[center] + new_center  
            node_features = F.relu(updates_node)
                
            # updates_node = []
            # for update in updates:
            #     updates_node.append(torch.stack(update))
                
            # node_features = F.relu(torch.vstack(updates_node))
        
        return self.final(node_features)