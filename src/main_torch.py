import os
import torch
import matplotlib.pyplot as plt
import argparse
from torch import nn, optim
import numpy as np

from utils import train_graph, test_graph, EarlyStopping
from data import load_dataset, eval_dataset, random_split
from model_torch import QGNNGraphClassifier
from test import HandcraftGNN, HandcraftGNN_NodeClassification

from datetime import datetime
import time

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

result_dir = os.path.join('../results')
os.makedirs(result_dir, exist_ok=True)
os.makedirs(os.path.join(result_dir, 'fig'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'log'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'model'), exist_ok=True)

param_file = os.path.join(result_dir, 'log', f"{timestamp}_model_parameters.txt")
grad_file = os.path.join(result_dir, 'log', f"{timestamp}_model_gradients.txt")


def get_args():
    parser = argparse.ArgumentParser(description="Train QGNN on graph data")
    parser.add_argument('--dataset', type=str, default='MUTAG', help='Dataset name (e.g., MUTAG, ENZYMES, CORA)')
    parser.add_argument('--train_size', type=float, default=0.7)
    parser.add_argument('--eval_size', type=float, default=0.5)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-2)
    parser.add_argument('--w_decay', type=float, default=0)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--node_qubit', type=int, default=3)
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--num_ent_layers', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1712)
    parser.add_argument('--task', type=str, default='graph', choices=['graph', 'node'], help='graph or node classification')

    parser.add_argument('--plot', action='store_true', help='Enable plotting')
    parser.add_argument('--save_model', action='store_true', help='Enable saving model')
    parser.add_argument('--gradient', action='store_true', help='Enable gradient saving')
    parser.add_argument('--results', action='store_true', help='Evaluate results')
    parser.add_argument('--model', type=str, default='qgnn',
                        choices=['qgnn', 'handcraft', 'gin', 'gcn', 'gat', 'sage', 'trans'],
                        help="Which model to run"
                        )
    parser.add_argument('--graphlet_size', type=int, default=10)
    parser.add_argument('--criterion', type=str, default='crossentropy',
                        choices=['crossentropy', 'MSE', 'BCE', 'L1'],
                        help="Which loss function to train model")
    return parser.parse_args()


def main(args):
    args.node_qubit = args.graphlet_size
    edge_qubit = args.node_qubit - 1
    n_qubits = args.node_qubit + edge_qubit
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_dev = None  
    w_shapes_dict = {
        'spreadlayer': (0, n_qubits, 1),
        'strong': (2, args.num_ent_layers, 3, 3),
        'inits': (1, 4),
        'update': (1, args.num_ent_layers, 3, 3),
        'twodesign': (0, args.num_ent_layers, 1, 2)
    }
    dataset, train_loader, test_loader, task_type = load_dataset(
        name=args.dataset,
        path='../data',
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size
    )
    node_input_dim = dataset[0].x.shape[1] if dataset[0].x is not None else 0
    edge_input_dim = dataset[0].edge_attr.shape if dataset[0].edge_attr is not None else 0
    if edge_input_dim:
        edge_input_dim = edge_input_dim[0] if len(edge_input_dim) < 2 else edge_input_dim[1]
    if args.dataset == "ZINC":
        num_classes = 1
    else:
        num_classes = dataset.num_classes
    if args.task == 'graph':
        if args.model == 'qgnn':
            model = QGNNGraphClassifier(
                node_input_dim=node_input_dim,
                edge_input_dim=edge_input_dim,
                graphlet_size=args.node_qubit,       
                hop_neighbor=args.num_gnn_layers,
                num_classes=num_classes,
                one_hot=0,
                # entangling_depth=args.num_ent_layers   
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
        elif args.model == 'gin':
            from baseline import GIN_Graph
            model = GIN_Graph(
                in_channels=node_input_dim,
                hidden_channels=args.hidden_channels,
                out_channels=num_classes,
                num_layers=args.num_gnn_layers,
            )
        elif args.model == 'gcn':
            from baseline import GCN_Graph
            model = GCN_Graph(
                in_channels=node_input_dim,
                hidden_channels=args.hidden_channels,
                out_channels=num_classes,
                num_layers=args.num_gnn_layers,
            )
        elif args.model == 'gat':
            from baseline import GAT_Graph
            model = GAT_Graph(
                in_channels=node_input_dim,
                hidden_channels=args.hidden_channels // 8,
                out_channels=num_classes,
                num_layers=args.num_gnn_layers,
                heads=8,
            )
        elif args.model == 'sage':
            from baseline import GraphSAGE_Graph
            model = GraphSAGE_Graph(
                in_channels=node_input_dim,
                hidden_channels=args.hidden_channels,
                out_channels=num_classes,
                num_layers=args.num_gnn_layers
            )
        elif args.model == 'trans':
            from baseline import Transformer_Graph
            model = Transformer_Graph(
                in_channels=node_input_dim,
                hidden_channels=args.hidden_channels // 8,
                out_channels=num_classes,
                num_layers=args.num_gnn_layers,
                heads=8,
            )
        else:
            raise ValueError(f"Unsupported model for graph task: {args.model}")

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
                hidden_channels=args.hidden_channels,
                out_channels=num_classes,
                num_layers=args.num_gnn_layers,
            )
        elif args.model == 'gcn':
            from baseline import GCN_Node
            model = GCN_Node(
                in_channels=node_input_dim,
                hidden_channels=args.hidden_channels,
                out_channels=num_classes,
                num_layers=args.num_gnn_layers,
            )
        elif args.model == 'gat':
            from baseline import GAT_Node
            model = GAT_Node(
                in_channels=node_input_dim,
                hidden_channels=8,
                out_channels=num_classes,
                num_layers=args.num_gnn_layers,
                heads=8,
            )
        elif args.model == 'sage':
            from baseline import GraphSAGE_Node
            model = GraphSAGE_Node(
                in_channels=dataset.num_features,
                hidden_channels=args.hidden_channels,
                out_channels=dataset.num_classes,
                num_layers=args.num_gnn_layers
            )
        else:
            raise ValueError(f"Unsupported model for node task: {args.model}")
    else:
        raise ValueError("Unsupported task type")

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    if args.criterion == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'MSE':
        criterion = nn.MSELoss()
    elif args.criterion == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    elif args.criterion == 'L1':
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unssuported loss function")

    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    if args.gradient:
        string = "="*10 + f"{timestamp}_{args.model}_{args.graphlet_size}_{args.dataset.lower()}_{args.epochs}epochs_lr{args.lr}_{args.gamma}over{args.step_size}" + "="*10
        with open(param_file, "w") as f_param:
            f_param.write(string + "\n")
        with open(grad_file, "w") as f_grad:
            f_grad.write(string + "\n")

    start = time.time()
    step_plot = args.epochs // 10 if args.epochs > 10 else 1

    model_save = os.path.join(result_dir, 'model', f"{timestamp}_{args.model}_{args.dataset.lower()}.pt")
    early_stopping = EarlyStopping(patience=10, save_path=model_save)

    print(f"{timestamp} \n")
    print(f"Training model {args.model} on {args.dataset} with {args.graphlet_size} graphlet size with {args.epochs} epochs, "
          f"learning rate {args.lr}, step size {args.step_size}, and gamma {args.gamma}.")

    if args.task == 'graph':
        for epoch in range(1, args.epochs + 1):
            train_graph(model, optimizer, train_loader, criterion, device)
            train_loss, train_acc, f1_train = test_graph(model, train_loader, criterion, device, num_classes)
            test_loss, test_acc, f1_test = test_graph(model, test_loader, criterion, device, num_classes)
            scheduler.step()

            if args.save_model:
                early_stopping(test_loss, model)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if args.gradient:
                with open(param_file, "a") as f_param:
                    f_param.write("="*40 + f" Epoch {epoch} " + "="*40 + "\n")
                    for name, param in model.named_parameters():
                        f_param.write(f"{name}:\n{param.data.detach().cpu().numpy()}\n\n")

                with open(grad_file, "a") as f_grad:
                    f_grad.write("="*40 + f" Epoch {epoch} " + "="*40 + "\n")
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            if param.grad is None:
                                f_grad.write(f"{name}: No gradient (None)\n")
                            else:
                                grad = param.grad.detach().cpu().numpy()
                                f_grad.write(f"{name}:\n{grad}\n\n")

            if epoch % step_plot == 0:
                print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.gamma,
            patience=args.epochs // 10 if args.epochs > 10 else 1,
        )
        from utils import train_node, test_node
        data = dataset[0].to(device)
        for epoch in range(1, args.epochs + 1):
            train_loss = train_node(model, optimizer, data, criterion, device)
            test_metrics = test_node(model, data, criterion, device, num_classes)

            train_losses.append(test_metrics['train']['loss'])
            test_losses.append(test_metrics['test']['loss'])
            train_accs.append(test_metrics['train']['acc'])
            test_accs.append(test_metrics['test']['acc'])

            if args.save_model:
                early_stopping(test_losses[-1], model)

            scheduler.step(test_metrics['val']['loss'])
            if epoch % step_plot == 0:
                print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} |"
                      f"Train Acc: {test_metrics['train']['acc']:.4f} | "
                      f"Val Acc: {test_metrics['val']['acc']:.4f} | Test Acc: {test_metrics['test']['acc']:.4f}")

    end = time.time()
    print(f"Total execution time: {end - start:.6f} seconds")

    if args.plot:
        epochs_range = range(1, args.epochs + 1)

        plt.figure(figsize=(10, 5))
        plt.suptitle(f"{args.model.upper()} on {args.dataset.upper()}", fontsize=14)
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

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path = f"plot_{timestamp}_{args.model}_{args.graphlet_size}_{args.dataset.lower()}_{args.epochs}epochs_lr{args.lr}_{args.gamma}over{args.step_size}.png"
        plt.savefig(os.path.join('../results/fig', plot_path), dpi=300)

    if args.results:
        accuracies = []
        num_runs = 100
        for each in range(num_runs):
            eval_loader = eval_dataset(
                name=args.dataset,
                path='../data',
                eval_size=args.eval_size,
                batch_size=args.batch_size,
                seed=args.seed + each
            )
            if args.task == 'graph':
                _, eval_acc, _ = test_graph(model, eval_loader, criterion, device, num_classes)
            elif args.task == 'node':
                eval_loader = random_split(eval_loader, train_ratio=0.2, val_ratio=0.5, seed=args.seed + each)
                from utils import test_node
                eval_metrics = test_node(model, eval_loader, criterion, device, num_classes)
                eval_acc = eval_metrics['val']['acc']
            else:
                raise ValueError(f"Unsupported task: {args.task}")

            accuracies.append(eval_acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1)
        print(f"{args.model} Mean Accuracy: {mean_acc:.4f} ± {std_acc:.3f}")


if __name__ == "__main__":
    args = get_args()
    main(args)
