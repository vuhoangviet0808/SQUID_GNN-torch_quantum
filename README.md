# Quantum Graph Neural Network (QGNN)

This project implements a Quantum-Inspired Graph Neural Network using PennyLane and PyTorch Geometric. It supports graph classification on datasets like MUTAG, ENZYMES, and others.

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages:
- torch==2.7.0
- torch-geometric==2.6.1
- pennylane==0.38.0

---

## 🚀 Training a QGNN

Run training with:
```bash
!python main.py --task graph --dataset MUTAG  --epochs 50 --lr 0.005 --step_size 10 --gamma 0.9\
--train_size 100 --test_size 50 --eval_size 100 --seed 1309\
--model gin --num_gnn_layers 3 --hidden_channels 128  \
--results --plot --gradient --save_model 
```

### Parameters

| Argument              | Description                                | Default |
|-----------------------|--------------------------------------------|---------|
| `--dataset`           | Dataset name (MUTAG, ENZYMES, CORA...)     | MUTAG   |
| `--train_size`        | Number of training samples (graph-level)   | 100     |
| `--test_size`         | Number of test samples                     | 50      |
| `--batch_size`        | Batch size for DataLoader                  | 32      |
| `--epochs`            | Number of training epochs                  | 20      |
| `--lr`                | Learning rate                              | 0.05    |
| `--node_qubit`        | Number of qubits per graphlet node         | 3       |
| `--num_qgnn_layers`   | QGNN message passing steps                 | 2       |
| `--num_ent_layers`    | Depth of entangling layers                 | 2       |

---

## 🧠 Model Overview

The model includes:

1. Graph decomposition into substructures
2. Quantum circuit-based message passing using PennyLane
3. Final embedding aggregation for classification

All quantum circuits are built with `qml.StronglyEntanglingLayers`, and trained using classical optimizers.

---

## 🗃️ Dataset Support

Currently supported:
- MUTAG
- ENZYMES
- PROTEINS
- CORA *(planned for node classification)*

Add new datasets via `data.py`.

---

## 📁 Repository Structure
```
├── main.py # Entry point with training loop \br
├── model.py # QGNN model definition 
├── utils.py # Train/test utilities 
├── data.py # Dataset loader (graph/node classification) 
├── README.md 
├── .gitignore 
└── requirements.txt 
```
---

## 📬 Contact
Maintained by tung.giangle99@gmail.com