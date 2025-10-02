import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy as sp
import numpy as np

from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
from tensorflow.keras.activations import linear
from tensorflow.keras import regularizers
from utils import star_subgraph


def input_process(x):
    return tf.math.tanh(x) * np.pi

def build_ring_cz_entangler(qubits):
    ops = []
    n = len(qubits)
    for i in range(n):
        a = qubits[i]
        b = qubits[(i + 1) % n]
        ops.append(cirq.CZ(a, b))
    return ops

def controlled_rot(control, target, axis, symbol):
    if axis == 'x':
        return cirq.ControlledGate(cirq.rx(symbol))(control, target)
    if axis == 'y':
        return cirq.ControlledGate(cirq.ry(symbol))(control, target)
    if axis == 'z':
        return cirq.ControlledGate(cirq.rz(symbol))(control, target)
    raise ValueError("axis must be x/y/z")


# ====== Mạch “message passing PQC” ======
def build_message_passing_circuit(edge_q, center_q, neighbor_q, anc1_q, anc2_q,
                                  data_syms, weight_syms, block_id="0"):
    c = cirq.Circuit()

    phi0, phi1, phi2, phi3 = data_syms 
    c += controlled_rot(neighbor_q, anc1_q, 'x', phi0)
    c += controlled_rot(edge_q,     anc1_q, 'y', phi1)
    c += controlled_rot(neighbor_q, anc2_q, 'z', phi2)
    c += controlled_rot(edge_q,     anc2_q, 'y', phi3)
    qA = [edge_q, neighbor_q, anc1_q]
    for i, q in enumerate(qA):
        c += cirq.rx(weight_syms[f"a_rx_{block_id}_{i}"])(q)
        c += cirq.ry(weight_syms[f"a_ry_{block_id}_{i}"])(q)
        c += cirq.rz(weight_syms[f"a_rz_{block_id}_{i}"])(q)
    c += build_ring_cz_entangler(qA)
    qB = [anc1_q, neighbor_q, anc2_q]
    for i, q in enumerate(qB):
        c += cirq.rx(weight_syms[f"b_rx_{block_id}_{i}"])(q)
        c += cirq.ry(weight_syms[f"b_ry_{block_id}_{i}"])(q)
        c += cirq.rz(weight_syms[f"b_rz_{block_id}_{i}"])(q)
    c += build_ring_cz_entangler(qB)

    return c


def build_center_update_block(center_q, anc1_q, anc2_q, weight_syms, block_id="upd"):
    c = cirq.Circuit()
    qU = [center_q, anc1_q, anc2_q]
    for i, q in enumerate(qU):
        c += cirq.rx(weight_syms[f"u_rx_{block_id}_{i}"])(q)
        c += cirq.ry(weight_syms[f"u_ry_{block_id}_{i}"])(q)
        c += cirq.rz(weight_syms[f"u_rz_{block_id}_{i}"])(q)
    c += build_ring_cz_entangler(qU)
    return c
class StarSubgraphPQC(layers.Layer):

    def __init__(self, graphlet_size=4, name=None):
        super().__init__(name=name)
        self.graphlet_size = graphlet_size  
        self.num_edges = graphlet_size - 1
        self.num_nodes = graphlet_size
        self.n_qubits = (self.num_edges + self.num_nodes) + 2
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
        self.weight_symbols = []
        def add_block(prefix, n_qubits):
            for i in range(n_qubits):
                self.weight_symbols += [
                    sp.Symbol(f"{prefix}_rx_{i}"),
                    sp.Symbol(f"{prefix}_ry_{i}"),
                    sp.Symbol(f"{prefix}_rz_{i}")
                ]
        add_block("a", 3)
        add_block("b", 3)  
        add_block("u", 3)    
        circuit = cirq.Circuit()
        esyms_iter = iter(self.data_symbols[: 2*self.num_edges])
        for i, q in enumerate(self.edge_qs):
            ry_s = next(esyms_iter); rz_s = next(esyms_iter)
            circuit += cirq.ry(ry_s)(q)
            circuit += cirq.rz(rz_s)(q)
        n_start = 2*self.num_edges
        n_end = n_start + 2*self.num_nodes
        nsyms_iter = iter(self.data_symbols[n_start:n_end])
        for i, q in enumerate(self.node_qs):
            ry_s = next(nsyms_iter); rz_s = next(nsyms_iter)
            circuit += cirq.ry(ry_s)(q)
            circuit += cirq.rz(rz_s)(q)
        init_syms = self.data_symbols[n_end:n_end+4]
        wnames = [str(s) for s in self.weight_symbols]
        wdict = {}
        for s in self.weight_symbols:
            wdict[str(s)] = s
        c_msg = build_message_passing_circuit(
            edge_q=self.edge_qs[0],
            center_q=self.node_qs[self.center_idx],
            neighbor_q=self.node_qs[1],
            anc1_q=self.anc1,
            anc2_q=self.anc2,
            data_syms=init_syms,
            weight_syms={
                "a_rx_0_0": wdict["a_rx_0"], "a_ry_0_0": wdict["a_ry_0"], "a_rz_0_0": wdict["a_rz_0"],
                "a_rx_0_1": wdict["a_rx_1"], "a_ry_0_1": wdict["a_ry_1"], "a_rz_0_1": wdict["a_rz_1"],
                "a_rx_0_2": wdict["a_rx_2"], "a_ry_0_2": wdict["a_ry_2"], "a_rz_0_2": wdict["a_rz_2"],
                "b_rx_0_0": wdict["b_rx_0"], "b_ry_0_0": wdict["b_ry_0"], "b_rz_0_0": wdict["b_rz_0"],
                "b_rx_0_1": wdict["b_rx_1"], "b_ry_0_1": wdict["b_ry_1"], "b_rz_0_1": wdict["b_rz_1"],
                "b_rx_0_2": wdict["b_rx_2"], "b_ry_0_2": wdict["b_ry_2"], "b_rz_0_2": wdict["b_rz_2"],
            },
            block_id="0"
        )
        circuit += c_msg
        c_upd = build_center_update_block(
            self.node_qs[self.center_idx], self.anc1, self.anc2,
            {
                "u_rx_upd_0": wdict["u_rx_0"], "u_ry_upd_0": wdict["u_ry_0"], "u_rz_upd_0": wdict["u_rz_0"],
                "u_rx_upd_1": wdict["u_rx_1"], "u_ry_upd_1": wdict["u_ry_1"], "u_rz_upd_1": wdict["u_rz_1"],
                "u_rx_upd_2": wdict["u_rx_2"], "u_ry_upd_2": wdict["u_ry_2"], "u_rz_upd_2": wdict["u_rz_2"],
            },
            block_id="upd"
        )
        circuit += c_upd
        readout = cirq.X(self.node_qs[self.center_idx])
        self.circuit = circuit
        self.readout = [readout]
        self.pqc = tfq.layers.PQC(self.circuit, self.readout, differentiator=tfq.differentiators.Adjoint())

        # Lưu thứ tự symbol cho dữ liệu
        self.data_symbol_names = [str(s) for s in self.data_symbols]

    def call(self, data_symbol_values):
        batch_size = tf.shape(data_symbol_values)[0]
        circuits = tfq.convert_to_tensor([self.circuit] * batch_size)
        return self.pqc([circuits, data_symbol_values])


class QGNNGraphClassifierTFQ(Model):
    def __init__(self,
                 node_input_dim=1,
                 edge_input_dim=1,
                 graphlet_size=4,
                 hop_neighbor=1,
                 num_classes=2,
                 hidden_dim=128,
                 dropout=0.3,
                 one_hot=False,
                 name=None):
        super().__init__(name=name)
        self.graphlet_size = graphlet_size
        self.hop_neighbor = hop_neighbor
        self.one_hot = one_hot
        self.hidden_dim = hidden_dim

        self.pqc_out_dim = 1   
        self.per_node_dim = 2  
        self.per_edge_dim = 2

        if self.one_hot:
            node_input_dim = 1
            edge_input_dim = 1
        else:
            edge_input_dim = max(edge_input_dim, 1)
        self.input_node = tf.keras.Sequential([
            Dense(hidden_dim, activation='leaky_relu'),
            Dropout(dropout),
            Dense(self.per_node_dim, activation=None)
        ])

        self.input_edge = tf.keras.Sequential([
            Dense(hidden_dim, activation='leaky_relu'),
            Dropout(dropout),
            Dense(self.per_edge_dim, activation=None)
        ])
        self.star_pqc_layers = [StarSubgraphPQC(graphlet_size=graphlet_size, name=f"star_pqc_{i+1}")
                                for i in range(hop_neighbor)]
        self.updates_ = [
            tf.keras.Sequential([
                Dense(hidden_dim, activation='leaky_relu'),
                Dropout(dropout),
                Dense(self.per_node_dim, activation=None)
            ]) for _ in range(hop_neighbor)
        ]

        self.norms = [LayerNormalization(axis=-1) for _ in range(hop_neighbor)]

        self.graph_head = tf.keras.Sequential([
            Dense(num_classes, activation='leaky_relu'),
            Dropout(dropout),
            Dense(num_classes, activation=None)
        ])

    def call(self, node_feat, edge_attr, edge_index, batch, training=False):
        if edge_attr is None:
            edge_attr = tf.ones((tf.shape(edge_index)[1], 1), dtype=node_feat.dtype)

        edge_index = tf.transpose(edge_index)
        num_nodes = tf.shape(node_feat)[0]

        node_f = tf.cast(node_feat, tf.float32)
        edge_f = tf.cast(edge_attr, tf.float32)

        node_f = self.input_node(node_f, training=training)
        edge_f = self.input_edge(edge_f, training=training)

        node_f = input_process(node_f)
        edge_f = input_process(edge_f)
        E = tf.shape(edge_index)[0]
        adj = tf.zeros((num_nodes, num_nodes), dtype=tf.int32)
        adj = tf.tensor_scatter_nd_update(adj, edge_index, tf.ones((E,), dtype=tf.int32))
        adj = tf.maximum(adj, tf.transpose(adj))
        subgraphs = star_subgraph(adj.numpy(), subgraph_size=self.graphlet_size)
        ei = edge_index.numpy()
        idx_dict = {(int(min(u, v)), int(max(u, v))): i for i, (u, v) in enumerate(ei)}
        for hop in range(self.hop_neighbor):
            pqc_layer = self.star_pqc_layers[hop]
            upd_layer = self.updates_[hop]
            norm_layer = self.norms[hop]

            centers = []
            updates = []

            for sub in subgraphs:
                center, *neighbors = sub
                n_feat = tf.gather(node_f, sub)  
                edge_idxs = [idx_dict[(min(center, int(n)), max(center, int(n)))] for n in neighbors]
                e_feat = tf.gather(edge_f, edge_idxs)  
                data_list = []
                for i in range(self.graphlet_size - 1):
                    data_list.append(e_feat[i, 0]) 
                    data_list.append(e_feat[i, 1]) 
                for i in range(self.graphlet_size):
                    data_list.append(n_feat[i, 0])  
                    data_list.append(n_feat[i, 1])  
                phi0 = n_feat[1, 0]  
                phi1 = e_feat[0, 0]
                phi2 = n_feat[1, 1]
                phi3 = e_feat[0, 1]
                data_list += [phi0, phi1, phi2, phi3]

                data_vals = tf.stack(data_list)[tf.newaxis, :] 

           
                exp = pqc_layer(data_vals) 
                exp = tf.squeeze(exp, axis=0)
                upd_inp = tf.concat([node_f[center], tf.reshape(exp, [1])], axis=0)  
                upd_vec = upd_layer(tf.expand_dims(upd_inp, 0), training=training)   
                updates.append(tf.squeeze(upd_vec, axis=0))
                centers.append(center)

            centers = tf.constant(centers, dtype=tf.int32)
            updates = tf.stack(updates, axis=0) 
            updates_node = tf.zeros_like(node_f)
            updates_node = tf.tensor_scatter_nd_add(
                updates_node,
                indices=tf.expand_dims(centers, axis=1),
                updates=updates
            )
            node_f = node_f + updates_node
        num_graphs = tf.reduce_max(batch) + 1
        graph_embedding = tf.math.unsorted_segment_mean(node_f, batch, num_graphs)

        logits = self.graph_head(graph_embedding, training=training)
        return logits
