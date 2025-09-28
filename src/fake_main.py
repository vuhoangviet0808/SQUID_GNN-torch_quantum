import os
import torch
import matplotlib.pyplot as plt
import pennylane as qml
import argparse
from torch import nn, optim


from utils import train_graph, test_graph
from data import load_dataset
from model import QGNNGraphClassifier
from test import HandcraftGNN, HandcraftGNN_NodeClassification

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")


def get_args():
    parser = argparse.ArgumentParser(description="Train QGNN on graph data")
    parser.add_argument('--dataset', type=str, default='MUTAG', help='Dataset name (e.g., MUTAG, ENZYMES, CORA)')
    parser.add_argument('--train_size', type=int, default=100)
    parser.add_argument('--test_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-2)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--node_qubit', type=int, default=3)
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--num_ent_layers', type=int, default=2)
    parser.add_argument('--task', type=str, default='graph', choices=['graph', 'node'], help='graph or node classification')

    
    # Debug options
    parser.add_argument('--plot', action='store_true', help='Enable plotting')
    
    # For switching between models
    parser.add_argument('--model', type=str, default='qgnn', 
                        choices=['qgnn', 'handcraft', 'gin', 'gcn', 'gat'],
                        help="Which model to run"
                        )
    parser.add_argument('--graphlet_size', type=int, default=10)
    
    
    return parser.parse_args()


def main(args):
    edge_qubit = args.node_qubit - 1
    n_qubits = args.node_qubit + edge_qubit
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_dev = qml.device("default.qubit", wires=n_qubits)

    # PQC weight shape settings
    w_shapes_dict = {
        'spreadlayer': (2, n_qubits, 1),
        'strong': (1, args.num_ent_layers, 3, 3),
        'inits': (0, 2),
        'twodesign': (0, args.num_ent_layers, 1, 2)
    }

    # Load dataset
    dataset, train_loader, test_loader, task_type = load_dataset(
        name=args.dataset,
        path='../data',
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size
    )

    # if task_type != 'graph':
    #     raise NotImplementedError("Node classification support is not implemented yet.")
 
    # Model metadata
    node_input_dim = dataset[0].x.shape[1] if dataset[0].x is not None else 0
    edge_input_dim = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 0
    num_classes = dataset.num_classes
    # Model init
    if args.task == 'graph':
        if args.model == 'qgnn':
            model = QGNNGraphClassifier(
                q_dev=q_dev,
                w_shapes=w_shapes_dict,
                node_input_dim=node_input_dim,
                edge_input_dim=edge_input_dim,
                graphlet_size=args.node_qubit,
                hop_neighbor=args.num_gnn_layers,
                num_classes=num_classes,
                one_hot=0
            )
        elif args.model == 'handcraft':
            model = HandcraftGNN(
                q_dev=q_dev,
                w_shapes=w_shapes_dict,
                node_input_dim=node_input_dim,
                edge_input_dim=edge_input_dim,
                graphlet_size=args.graphlet_size,
                hop_neighbor=args.num_gnn_layers,
                num_classes=num_classes,
                one_hot=0
            )
        else:
            raise ValueError("Unknown model type: use 'qgnn' or 'handcraft'")
    elif args.task == 'node':
        data = dataset[0].to(device)
        if args.model == 'handcraft':
            model = HandcraftGNN_NodeClassification(
                q_dev=q_dev,
                w_shapes=w_shapes_dict,
                node_input_dim=node_input_dim,
                edge_input_dim=edge_input_dim,
                graphlet_size=args.graphlet_size,
                hop_neighbor=args.num_gnn_layers,
                num_classes=num_classes,
                one_hot=0
            )
        elif args.model == 'gin':
            from baseline import GIN_Node
            model = GIN_Node(
                in_channels=node_input_dim,
                hidden_channels=64,
                out_channels=num_classes,
                num_layers=args.num_gnn_layers,
            )
        elif args.model == 'gcn':
            from baseline import GCN_Node
            model = GCN_Node(
                in_channels=node_input_dim,
                hidden_channels=64,
                out_channels=num_classes,
                num_layers=args.num_gnn_layers,
            )
        elif args.model == 'gat':
            from baseline import GAT_Node
            model = GAT_Node(
                in_channels=node_input_dim,
                hidden_channels=8,    # heads * hidden
                out_channels=num_classes,
                num_layers=args.num_gnn_layers,
                heads=8,
            )
        else:
            raise ValueError(f"Unsupported model for node task: {args.model}")
    else:
        raise ValueError("Unsupported task type")
    
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()

    ## Note: For debugging purposes, you can uncomment the following lines to print model details. 
    # ##
    # print("=" * 50)
    # print(f"Training on dataset: {args.dataset.upper()}")
    # print(f"Node feature dimension: {node_input_dim}")
    # print(f"Edge feature dimension: {edge_input_dim}")
    # print(f"Number of classes: {num_classes}")
    # print(f"Number of training samples: {len(train_loader.dataset)}")
    # print(f"Number of testing samples: {len(test_loader.dataset)}")
    # print(f"QGNN layers: {args.num_gnn_layers}")
    # print(f"Entangling layers per PQC: {args.num_ent_layers}")
    # print(f"Total qubits: {n_qubits} (Node qubits: {args.node_qubit}, Edge qubits: {edge_qubit})")
    # print(f"Epochs: {args.epochs}")
    # print(f"Batch size: {args.batch_size}")
    # print(f"Learning rate: {args.lr}")
    # print("=" * 50)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    # Training loop
    step_plot = args.epochs // 10 if args.epochs > 10 else 1
    if args.task == 'graph':
        for epoch in range(1, args.epochs + 1):
            train_graph(model, optimizer, train_loader, criterion, device)
            train_loss, train_acc, f1_train = test_graph(model, train_loader, criterion, device, num_classes)
            test_loss, test_acc, f1_test = test_graph(model, test_loader, criterion, device, num_classes)
            scheduler.step()
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            if epoch % step_plot == 0:
                print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    else:  # node task
        from utils import train_node, test_node
        for epoch in range(1, args.epochs + 1):
            train_loss = train_node(model, optimizer, data, criterion, device)
            test_metrics = test_node(model, data, criterion, device, num_classes)
            train_losses.append(test_metrics['train']['loss'])
            test_losses.append(test_metrics['test']['loss'])
            train_accs.append(test_metrics['train']['acc'])
            test_accs.append(test_metrics['test']['acc'])
            # scheduler.step()
            if epoch % step_plot == 0:
                print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} |" +
                    f"Train Acc: {test_metrics['train']['acc']:.4f} | "
                    f"Val Acc: {test_metrics['val']['acc']:.4f} | Test Acc: {test_metrics['test']['acc']:.4f}")
    
    
    if args.plot:
        epochs_range = range(1, args.epochs + 1)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label="Train Loss")
        plt.plot(epochs_range, test_losses, label="Test Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_accs, label="Train Acc")
        plt.plot(epochs_range, test_accs, label="Test Acc")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        # plot_path = f"plot_{args.model}_{args.graphlet_size}_{args.dataset.lower()}_{args.epochs}epochs_lr{args.lr}_{args.gamma}over{args.step_size}.png"
        plot_path = f"plot_{timestamp}_{args.model}_{args.graphlet_size}_{args.dataset.lower()}_{args.epochs}epochs_lr{args.lr}_{args.gamma}over{args.step_size}.png"
        plt.savefig(os.path.join('../results', plot_path), dpi=300)

if __name__ == "__main__":
    args = get_args()
    main(args)
