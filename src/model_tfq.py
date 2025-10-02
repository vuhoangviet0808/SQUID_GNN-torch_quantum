# model_tfq.py
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy as sp
import numpy as np

from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
from torch_geometric.nn import global_mean_pool  
from utils import star_subgraph

def controlled_rot(control, target, axis, symbol):
    if axis == 'x': return cirq.ControlledGate(cirq.rx(symbol))(control, target)
    if axis == 'y': return cirq.ControlledGate(cirq.ry(symbol))(control, target)
    if axis == 'z': return cirq.ControlledGate(cirq.rz(symbol))(control, target)
    raise ValueError("axis must be 'x'|'y'|'z'.")

def entangle_block(qubits, prefix):
    ops = []
    for i, q in enumerate(qubits):
        ops += [cirq.rx(sp.Symbol(f"{prefix}_rx_{i}"))(q),
                cirq.ry(sp.Symbol(f"{prefix}_ry_{i}"))(q),
                cirq.rz(sp.Symbol(f"{prefix}_rz_{i}"))(q)]
    for i in range(len(qubits)):
        a, b = qubits[i], qubits[(i + 1) % len(qubits)]
        ops.append(cirq.CZ(a, b))
    return ops


class StarPQC(layers.Layer):
    def __init__(self, graphlet_size=4, name=None):
        super().__init__(name=name)
        self.k = graphlet_size
        self.num_edges = self.k - 1
        self.num_nodes = self.k

        # layout qubits: edges (row 0), nodes (row 1), ancilla (row 2: 0,1)
        self.edge_qs = [cirq.GridQubit(0, i) for i in range(self.num_edges)]
        self.node_qs = [cirq.GridQubit(1, i) for i in range(self.num_nodes)]
        self.anc1 = cirq.GridQubit(2, 0)
        self.anc2 = cirq.GridQubit(2, 1)
        self.center_idx = 0

        self.data_symbols = []
        for i in range(self.num_edges):
            self.data_symbols += [sp.Symbol(f"edge_{i}_ry"), sp.Symbol(f"edge_{i}_rz")]
        for i in range(self.num_nodes):
            self.data_symbols += [sp.Symbol(f"node_{i}_ry"), sp.Symbol(f"node_{i}_rz")]
        self.data_symbols += [sp.Symbol(f"init_phi_{j}") for j in range(4)]

        # circuit
        c = cirq.Circuit()

        # Encode edges
        it = iter(self.data_symbols[: 2 * self.num_edges])
        for q in self.edge_qs:
            c += [cirq.ry(next(it))(q), cirq.rz(next(it))(q)]

        # Encode nodes
        s = 2 * self.num_edges
        itn = iter(self.data_symbols[s: s + 2 * self.num_nodes])
        for q in self.node_qs:
            c += [cirq.ry(next(itn))(q), cirq.rz(next(itn))(q)]

        # Inits
        phi0, phi1, phi2, phi3 = self.data_symbols[s + 2 * self.num_nodes: s + 2 * self.num_nodes + 4]

        # only-rotation đoạn message (thay qml.CRX/CRY/CRZ/CRY)
        # neighbor -> anc1 (CRX), edge -> anc1 (CRY), neighbor -> anc2 (CRZ), edge -> anc2 (CRY)
        neighbor_q = self.node_qs[1]
        edge_q = self.edge_qs[0] if self.edge_qs else self.node_qs[1]
        c += [controlled_rot(neighbor_q, self.anc1, 'x', phi0),
              controlled_rot(edge_q,    self.anc1, 'y', phi1),
              controlled_rot(neighbor_q, self.anc2, 'z', phi2),
              controlled_rot(edge_q,    self.anc2, 'y', phi3)]

        # StronglyEntanglingLayers thay bằng 2 block xoay + CZ ring
        c += entangle_block([edge_q, neighbor_q, self.anc1], "A")
        c += entangle_block([self.anc1, neighbor_q, self.anc2], "B")

        # Update block trên (center, anc1, anc2)
        c += entangle_block([self.node_qs[self.center_idx], self.anc1, self.anc2], "U")

        self.readout = [cirq.X(self.node_qs[self.center_idx])]
        self.circuit = c
        weight_syms = []
        for prefix, n_q in [("A", 3), ("B", 3), ("U", 3)]:
            for i in range(n_q):
                for ax in ("rx", "ry", "rz"):
                    weight_syms.append(f"{prefix}_{ax}_{i}")

        self.weight_symbol_names = weight_syms 
        # self.pqc = tfq.layers.PQC(self.circuit, self.readout)
        

        self.theta = self.add_weight(
            name="theta_pqc",
            shape=(len(self.weight_symbol_names),),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
            trainable=True,
        )
        self.expect = tfq.layers.Expectation()
        self.data_symbol_names = [str(s) for s in self.data_symbols] 
    def call(self, data_vals):
        circuits_1 = tfq.convert_to_tensor([self.circuit])  
        batch = tf.shape(data_vals)[0]
        circuits = tf.repeat(circuits_1, repeats=tf.cast(batch, tf.int32), axis=0) 
        symbol_names = tf.constant(self.data_symbol_names + self.weight_symbol_names, dtype=tf.string)
        data_vals = tf.cast(data_vals, tf.float32)  
        weight_vals = tf.tile(self.theta[tf.newaxis, :], [batch, 1])  
        symbol_values = tf.concat([data_vals, weight_vals], axis=1)  
        exp = self.expect(
            circuits,
            symbol_names=symbol_names,
            symbol_values=symbol_values,
            operators=self.readout,  
        ) 

        return tf.expand_dims(exp, axis=-1)
class QGNNGraphClassifierTFQ(Model):
    def __init__(self, q_dev=None, w_shapes=None, node_input_dim=1, edge_input_dim=1,
                 graphlet_size=4, hop_neighbor=1, num_classes=2, one_hot=0, hidden_dim=128, dropout=0.3, name=None):
        super().__init__(name=name)
        self.graphlet_size = graphlet_size
        self.hop_neighbor = hop_neighbor
        self.one_hot = one_hot
        self.pqc_dim = 2
        self.pqc_out = 1
        self.final_dim = self.pqc_dim

        # projector (giữ đúng kích thước đích 2)
        self.input_node = tf.keras.Sequential([
            Dense(hidden_dim, activation='leaky_relu'),
            Dropout(dropout),
            Dense(self.pqc_dim, activation=None)
        ])
        self.input_edge = tf.keras.Sequential([
            Dense(hidden_dim, activation='leaky_relu'),
            Dropout(dropout),
            Dense(self.pqc_dim, activation=None)
        ])

        # PQC & update per hop
        self.stars = [StarPQC(graphlet_size=graphlet_size, name=f"star_pqc_{i+1}") for i in range(hop_neighbor)]
        self.upds  = [tf.keras.Sequential([
                        Dense(hidden_dim, activation='leaky_relu'),
                        Dropout(dropout),
                        Dense(self.pqc_dim, activation=None)
                      ]) for _ in range(hop_neighbor)]
        self.norms = [LayerNormalization(axis=-1) for _ in range(hop_neighbor)]

        # head
        self.graph_head = tf.keras.Sequential([
            Dense(num_classes, activation='leaky_relu'),
            Dropout(dropout),
            Dense(num_classes, activation=None)
        ])

    @staticmethod
    def _input_process(x):
        return tf.math.tanh(x) * np.pi

    def call(self, node_feat, edge_attr, edge_index, batch, training=False):
        # Chuẩn hoá đầu vào giống bản gốc
        if edge_attr is None:
            edge_attr = tf.ones((tf.shape(edge_index)[1], 1), dtype=node_feat.dtype)

        node_f = tf.cast(node_feat, tf.float32)  # [N, F]
        edge_f = tf.cast(edge_attr, tf.float32)  # [E, Fe]
        ei = tf.transpose(edge_index)           # [E, 2]
        num_nodes = tf.shape(node_f)[0]

        node_f = self._input_process(self.input_node(node_f, training=training))  # -> [N,2]
        edge_f = self._input_process(self.input_edge(edge_f, training=training))  # -> [E,2]

        # adj & edge index dict như bản gốc
        E = tf.shape(ei)[0]
        adj = tf.zeros((num_nodes, num_nodes), dtype=tf.int32)
        adj = tf.tensor_scatter_nd_update(adj, ei, tf.ones((E,), dtype=tf.int32))
        adj = tf.maximum(adj, tf.transpose(adj))

        # dict (min(u,v), max(u,v)) -> idx
        ei_np = ei.numpy()
        idx_dict = {(int(min(u, v)), int(max(u, v))): i for i, (u, v) in enumerate(ei_np)}

        for hop in range(self.hop_neighbor):
            star = self.stars[hop]
            upd_layer = self.upds[hop]
            # norm_layer = self.norms[hop]

            subgraphs = star_subgraph(adj.numpy(), subgraph_size=self.graphlet_size)
            centers = []
            updates = []

            for sub in subgraphs:
                center, *neighbors = sub
                k = self.graphlet_size
                neighbors = neighbors[: max(0, k - 1)]
                deg = len(neighbors)

                # Node features (center + neighbors), pad tới k
                n_feat_raw = tf.gather(node_f, [center] + neighbors)  # [1+deg,2]
                if 1 + deg < k:
                    pad_nodes = tf.zeros([k - (1 + deg), tf.shape(n_feat_raw)[1]], dtype=n_feat_raw.dtype)
                    n_feat_pad = tf.concat([n_feat_raw, pad_nodes], axis=0)  # [k,2]
                else:
                    n_feat_pad = n_feat_raw  # [k,2]

                # Edge features (center-neighbor), pad tới k-1
                e_rows = []
                for n in neighbors:
                    key = (min(int(center), int(n)), max(int(center), int(n)))
                    if key in idx_dict:
                        e_rows.append(edge_f[idx_dict[key]])
                    else:
                        e_rows.append(tf.zeros([tf.shape(edge_f)[1]], dtype=edge_f.dtype))
                if deg < k - 1:
                    e_rows += [tf.zeros([tf.shape(edge_f)[1]], dtype=edge_f.dtype) for _ in range((k - 1) - deg)]
                e_feat_pad = tf.stack(e_rows, axis=0) if e_rows else tf.zeros([k - 1, tf.shape(edge_f)[1]], edge_f.dtype)

                # Pack data symbols: edges -> nodes -> inits(4)
                data_list = []
                for i in range(k - 1):
                    data_list += [e_feat_pad[i, 0], e_feat_pad[i, 1]]
                for i in range(k):
                    data_list += [n_feat_pad[i, 0], n_feat_pad[i, 1]]

                phi0 = n_feat_pad[1, 0] if deg >= 1 else tf.constant(0.0, dtype=node_f.dtype)
                phi1 = e_feat_pad[0, 0] if k - 1 >= 1 else tf.constant(0.0, dtype=edge_f.dtype)
                phi2 = n_feat_pad[1, 1] if deg >= 1 else tf.constant(0.0, dtype=node_f.dtype)
                phi3 = e_feat_pad[0, 1] if k - 1 >= 1 else tf.constant(0.0, dtype=edge_f.dtype)
                data_list += [phi0, phi1, phi2, phi3]

                data_vals = tf.expand_dims(tf.stack(data_list), axis=0)  # [1, n_syms]
                exp = star(data_vals)  # [1,1]
                exp = tf.squeeze(exp, axis=0)  # [1] -> scalar

                upd_inp = tf.concat([node_f[center], tf.reshape(exp, [1])], axis=0)  # [3]
                upd_vec = upd_layer(tf.expand_dims(upd_inp, 0), training=training)   # [1,2]
                updates.append(tf.squeeze(upd_vec, axis=0))
                centers.append(center)

            centers = tf.constant(centers, dtype=tf.int32)
            updates = tf.stack(updates, axis=0) if updates else tf.zeros_like(node_f[:0])
            updates_node = tf.zeros_like(node_f)
            updates_node = tf.tensor_scatter_nd_add(updates_node, tf.expand_dims(centers, 1), updates)

            node_f = node_f + updates_node  # residual
            # node_f = norm_layer(node_f, training=training)

        # dùng global_mean_pool của PyG (giữ nguyên như bản cũ)
        graph_embedding = global_mean_pool(torch_like(node_f), torch_like(batch))
        graph_embedding = tf.convert_to_tensor(graph_embedding.numpy(), dtype=tf.float32)

        return self.graph_head(graph_embedding, training=training)


# ===== nhỏ tiện ích để xài được global_mean_pool của PyG trên tensor TF =====
import torch
def torch_like(t):
    if isinstance(t, torch.Tensor):
        return t
    return torch.from_numpy(t.numpy()) if hasattr(t, "numpy") else torch.tensor(t)
