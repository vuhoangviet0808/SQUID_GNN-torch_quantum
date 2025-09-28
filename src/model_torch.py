# tq_layer.py
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.measurement import expval_joint_analytical
import torchquantum.functional as tqf
import torch.nn.functional as F
from torch_geometric.nn import MLP, global_add_pool, global_mean_pool, global_max_pool   
import math
from utils import star_subgraph


def _fetch_any_state_1d(qdev: torch.nn.Module) -> torch.Tensor:
    st = None
    for attr in ("states", "state", "_state"):
        st = getattr(qdev, attr, None)
        if st is not None:
            break
    if st is None:
        for m in ("get_states_1d", "get_states", "get_state"):
            fn = getattr(qdev, m, None)
            if callable(fn):
                st = fn()
                if st is not None:
                    break
    if st is None:
        raise RuntimeError("Không truy cập được state từ QuantumDevice")
    device = st.device
    if torch.is_complex(st):
        vec = st.reshape(-1)                 
        re = vec.real
        im = vec.imag
    else:
        t = st
        if t.dim() >= 2 and t.shape[0] > 1 and t.shape[-1] != 2:
            t = t[0]

        if t.dim() >= 1 and t.shape[-1] == 2:
            t2 = t.reshape(-1, 2)       
            re = t2[:, 0].contiguous()
            im = t2[:, 1].contiguous()
        else:
            # chỉ biên độ thực: im = 0
            vec = t.reshape(-1)
            re = vec.contiguous()
            im = torch.zeros_like(re, device=device)
    L = re.numel()
    if L == 0:
        raise RuntimeError("State rỗng.")
    n = int(math.floor(math.log2(L)))
    L2 = 1 << n
    if L2 != L:
        re = re[:L2]
        im = im[:L2]
    return re, im

def expval_x_scalar(qdev: torch.nn.Module, wire: int) -> torch.Tensor:
    tqf.hadamard(qdev, wires=wire)
    re, im = _fetch_any_state_1d(qdev)
    L = re.numel()
    device = re.device
    idx = torch.arange(L, device=device)
    bits = (idx >> wire) & 1
    sign = 1 - 2 * bits  
    probs = re.square() + im.square()
    z = (probs * sign).sum()            
    return z.reshape(1)
def small_normal_init(tensor):
    return nn.init.normal_(tensor, mean=0.0, std=0.1)

def uniform_pi_init(tensor):
    return nn.init.uniform_(tensor, a=0.0, b=torch.pi)

def identity_block_init(tensor):
    with torch.no_grad():
        tensor.zero_()
        if tensor.ndim < 1:
            return tensor
        total_params = tensor.numel()
        num_active = max(1, total_params // 3)
        flat = tensor.view(-1)
        idx = torch.randperm(flat.shape[0])[:num_active]
        flat[idx] = torch.randn_like(flat[idx]) * 0.1
        return tensor

def input_process(t):
    return torch.tanh(t) * torch.pi


class StrongBlock(nn.Module):
    def __init__(self, depth: int, n_qubits: int):
        super().__init__()
        assert n_qubits >= 2
        self.depth = depth
        self.nq = n_qubits
        # params trainable của block
        self.params = nn.Parameter(torch.zeros(depth, n_qubits, 3))
        nn.init.uniform_(self.params, a=0.0, b=torch.pi)

    def forward(self, qdev, wires):
        assert len(wires) == self.nq
        for d in range(self.depth):
            for qi, w in enumerate(wires):
                a = self.params[d, qi, 0].view(1, 1)  
                b = self.params[d, qi, 1].view(1, 1)  
                c = self.params[d, qi, 2].view(1, 1)  
                tqf.rz(qdev, wires=w, params=a)
                tqf.ry(qdev, wires=w, params=b)
                tqf.rx(qdev, wires=w, params=c)
            for i in range(self.nq - 1):
                tqf.cnot(qdev, wires=[wires[i], wires[i+1]])
            if self.nq >= 3:
                tqf.cnot(qdev, wires=[wires[-1], wires[0]])

class QMessagePassing(nn.Module):
    def __init__(self, graphlet_size: int, entangling_depth: int = 1):
        super().__init__()
        self.S = graphlet_size
        self.n_wires = 2 * self.S + 1
        self.anc1 = 2 * self.S - 1
        self.anc2 = 2 * self.S
        self.inits = nn.Parameter(torch.zeros(1, 4))
        nn.init.uniform_(self.inits, a=0.0, b=torch.pi)
        self.strong0 = StrongBlock(depth=entangling_depth, n_qubits=3)
        self.strong1 = StrongBlock(depth=entangling_depth, n_qubits=3)
        self.update  = StrongBlock(depth=max(1, entangling_depth // 1), n_qubits=3)

    def encode_features(self, qdev, adjacency_matrix, vertex_features):
        num_edges = adjacency_matrix.shape[0]
        for i in range(num_edges):
            a0, a1 = adjacency_matrix[i]
            # params shape nên là [batch, 1]; batch=1
            tqf.ry(qdev, wires=i, params=a0.view(1, 1))
            tqf.rz(qdev, wires=i, params=a1.view(1, 1))

        center_wire = self.S - 1
        num_nodes = vertex_features.shape[0]
        for i in range(num_nodes):
            v0, v1 = vertex_features[i]
            tqf.ry(qdev, wires=center_wire + i, params=v0.view(1, 1))
            tqf.rz(qdev, wires=center_wire + i, params=v1.view(1, 1))

    def message_block(self, qdev, edge, center, neighbor, anc1, anc2):
        tqf.crx(qdev, wires=[neighbor, anc1], params=self.inits[:, 0:1])
        tqf.cry(qdev, wires=[edge,    anc1], params=self.inits[:, 1:2])
        tqf.crz(qdev, wires=[neighbor, anc2], params=self.inits[:, 2:3])
        tqf.cry(qdev, wires=[edge,    anc2], params=self.inits[:, 3:4])
        self.strong0(qdev, wires=[edge, neighbor, anc1])
        self.strong1(qdev, wires=[anc1, neighbor, anc2])

    def forward(self, inputs_flat: torch.Tensor):
        feat = inputs_flat.view(-1, 2)
        num_edges = self.S - 1
        adjacency_matrix = feat[:num_edges, :]
        vertex_features = feat[num_edges:num_edges + self.S, :]
        qdev = tq.QuantumDevice(n_wires=self.n_wires)
        qdev.to(inputs_flat.device)

        self.encode_features(qdev, adjacency_matrix, vertex_features)

        center = self.S - 1
        anc1, anc2 = self.anc1, self.anc2

        for i in range(num_edges):
            edge = i
            neighbor = center + i + 1
            self.message_block(qdev, edge, center, neighbor, anc1, anc2)

        self.update(qdev, wires=[center, anc1, anc2])

        x_exp = expval_x_scalar(qdev, center)   
        return x_exp



class QGNNGraphClassifier(nn.Module):
    def __init__(
        self, 
        node_input_dim=1, edge_input_dim=1,
        graphlet_size=4, hop_neighbor=1, num_classes=2, one_hot=0
    ):
        super().__init__()
        self.hidden_dim = 128
        self.graphlet_size = graphlet_size
        self.one_hot = one_hot
        self.hop_neighbor = hop_neighbor
        self.pqc_dim  = 2          
        self.chunk    = 1
        self.final_dim = self.pqc_dim * self.chunk
        self.pqc_out  = 1
        if self.one_hot:
            self.node_input_dim = 1
            self.edge_input_dim = 1
        else:
            self.node_input_dim = node_input_dim
            self.edge_input_dim = max(1, edge_input_dim)
        self.input_node = MLP([self.node_input_dim, self.hidden_dim, self.final_dim],
                              act='leaky_relu', norm=None, dropout=0.3)
        self.input_edge = MLP([self.edge_input_dim, self.hidden_dim, self.pqc_dim],
                              act='leaky_relu', norm=None, dropout=0.3)

        self.qconvs = nn.ModuleDict()
        self.upds   = nn.ModuleDict()
        self.norms  = nn.ModuleDict()

        for i in range(self.hop_neighbor):
            self.qconvs[f"lay{i+1}"] = QMessagePassing(graphlet_size=self.graphlet_size)
            self.upds[f"lay{i+1}"] = MLP(
                [self.pqc_dim + self.pqc_out, self.hidden_dim, self.pqc_dim],
                act='leaky_relu', norm=None, dropout=0.3
            )
            self.norms[f"lay{i+1}"] = nn.LayerNorm(self.pqc_dim)

        self.graph_head = MLP([self.final_dim, num_classes, num_classes],
                              act='leaky_relu', norm=None, dropout=0.3)

    def forward(self, node_feat, edge_attr, edge_index, batch):
        edge_index = edge_index.t()
        num_nodes = node_feat.size(0)

        if edge_attr is None:
            edge_attr = torch.ones((edge_index.size(0), self.edge_input_dim), device=node_feat.device)

        e_feat = self.input_edge(edge_attr.float())
        n_feat = self.input_node(node_feat.float())

        n_feat = input_process(n_feat)
        e_feat = input_process(e_feat)
        idx_dict = {(min(int(u), int(v)), max(int(u), int(v))): i
                    for i, (u, v) in enumerate(edge_index.tolist())}
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.int, device='cpu')
        adj[edge_index[:,0].cpu(), edge_index[:,1].cpu()] = 1
        adj[edge_index[:,1].cpu(), edge_index[:,0].cpu()] = 1

        for i in range(self.hop_neighbor):
            subgraphs = star_subgraph(adj.numpy(), subgraph_size=self.graphlet_size)
            node_upd = torch.zeros((num_nodes, self.final_dim), device=n_feat.device)

            q_layer  = self.qconvs[f"lay{i+1}"]
            upd_layer = self.upds[f"lay{i+1}"]

            centers, updates = [], []
            for sub in subgraphs:
                center, *neighbors = sub

                n_local = n_feat[sub] 
                edge_idxs = [ idx_dict[(min(center, int(nb)), max(center, int(nb)))]
                              for nb in neighbors ]
                e_local = e_feat[edge_idxs]  
                inputs = torch.cat([e_local, n_local], dim=0).reshape(-1)
                all_msg = q_layer(inputs)          
                aggr = all_msg.reshape(-1)[:1]
                upd_vec = upd_layer(torch.cat([n_feat[center], aggr.to(n_feat.device)], dim=0))
                centers.append(center)
                updates.append(upd_vec)

            centers = torch.tensor(centers, device=n_feat.device, dtype=torch.long)
            updates = torch.stack(updates, dim=0)
            node_upd = node_upd.index_add(0, centers, updates)
            n_feat = n_feat + node_upd

        graph_embedding = global_mean_pool(n_feat, batch)
        return self.graph_head(graph_embedding)