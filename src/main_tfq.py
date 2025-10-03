import os
import argparse
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model_tfq import QGNNGraphClassifierTFQ
from data import load_dataset, eval_dataset, random_split

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

result_dir = os.path.join('../results')
os.makedirs(result_dir, exist_ok=True)
os.makedirs(os.path.join(result_dir, 'fig'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'log'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'model'), exist_ok=True)

param_file = os.path.join(result_dir, 'log', f"{timestamp}_model_parameters.txt")
grad_file  = os.path.join(result_dir, 'log', f"{timestamp}_model_gradients.txt")

def torch_batch_to_tf(batch):
    x = batch.x.cpu().numpy().astype(np.float32) if batch.x is not None else None
    ei = batch.edge_index.cpu().numpy().astype(np.int32)
    ea = batch.edge_attr.cpu().numpy().astype(np.float32) if batch.edge_attr is not None else None
    bvec = batch.batch.cpu().numpy().astype(np.int32)
    y = batch.y.cpu().numpy()
    if y.ndim == 1:
        y = y[:, None]

    if x is None:
        x = np.zeros((bvec.shape[0], 1), dtype=np.float32)

    return {
        "x": tf.convert_to_tensor(x),
        "edge_index": tf.convert_to_tensor(ei),
        "edge_attr": tf.convert_to_tensor(ea) if ea is not None else None,
        "batch": tf.convert_to_tensor(bvec),
        "y": tf.convert_to_tensor(y.astype(np.float32)),
    }


def make_tf_labels(y, num_classes, criterion):
    c = criterion.lower()
    if num_classes == 1:
        return tf.reshape(tf.cast(y, tf.float32), [-1, 1])
    if c == "crossentropy":
        return tf.reshape(tf.cast(y, tf.int32), [-1])
    elif c == "bce":
        return tf.reshape(tf.cast(y, tf.float32), [-1, 1])
    else:
        return tf.reshape(tf.cast(y, tf.float32), [-1, 1])


def make_loss(criterion, num_classes):
    c = criterion.lower()
    if num_classes == 1:
        if c == "mse":
            return tf.keras.losses.MeanSquaredError()
        elif c == "l1":
            return tf.keras.losses.MeanAbsoluteError()
        else:
            return tf.keras.losses.MeanSquaredError()
    else:
        if c == "crossentropy":
            return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        elif c == "bce":
            return tf.keras.losses.BinaryCrossentropy(from_logits=True)
        elif c == "mse":
            return tf.keras.losses.MeanSquaredError()
        elif c == "l1":
            return tf.keras.losses.MeanAbsoluteError()
        else:
            raise ValueError(f"Unsupported loss function: {criterion}")

def make_optimizer_and_schedule(lr, step_size, gamma, steps_per_epoch):
    decay_steps = max(1, int(step_size * max(1, steps_per_epoch)))
    schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=decay_steps,
        decay_rate=gamma,
        staircase=True,
    )
    opt = tf.keras.optimizers.Adam(learning_rate=schedule)
    return opt, schedule

def compute_weight_decay(model, w_decay):
    if w_decay <= 0:
        return 0.0
    l2 = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if "bias" not in v.name])
    return w_decay * l2

def train_step(model, optimizer, loss_fn, num_classes, criterion, w_decay, batch_tf):
    with tf.GradientTape() as tape:
        logits = model(batch_tf["x"], batch_tf["edge_attr"], batch_tf["edge_index"], batch_tf["batch"], training=True)
        y_true = make_tf_labels(batch_tf["y"], num_classes, criterion)
        base_loss = loss_fn(y_true, logits)
        reg_loss = compute_weight_decay(model, w_decay)
        loss = base_loss + reg_loss

        if num_classes == 1:
            base_loss = loss_fn(y_true, logits)
            acc = 0.0
        else:
            if criterion.lower() == "crossentropy":
                base_loss = loss_fn(y_true, logits)
                preds = tf.argmax(logits, axis=-1, output_type=tf.int32)  # [B]
                acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_true), tf.float32))
            elif criterion.lower() == "bce":
                base_loss = loss_fn(y_true, logits)
                preds = tf.cast(tf.sigmoid(logits) > 0.5, tf.float32)
                acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_true), tf.float32))
            else:
                base_loss = loss_fn(y_true, logits)
                acc = 0.0

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return float(loss.numpy()), float(acc.numpy()), grads

def eval_step(model, loss_fn, num_classes, criterion, w_decay, batch_tf):
    logits = model(batch_tf["x"], batch_tf["edge_attr"], batch_tf["edge_index"], batch_tf["batch"], training=False)
    y_true = make_tf_labels(batch_tf["y"], num_classes, criterion)
    base_loss = loss_fn(y_true, logits)
    reg_loss = compute_weight_decay(model, w_decay)
    loss = base_loss + reg_loss

    if num_classes == 1:
        acc = 0.0
    else:
        if criterion.lower() == "crossentropy":
            base_loss = loss_fn(y_true, logits)
            preds = tf.argmax(logits, axis=-1, output_type=tf.int32) 
            acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_true), tf.float32))
        elif criterion.lower() == "bce":
            base_loss = loss_fn(y_true, logits)
            preds = tf.cast(tf.sigmoid(logits) > 0.5, tf.float32)
            acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_true), tf.float32))
        else:
            acc = 0.0
    return float(loss.numpy()), float(acc.numpy())


def run_epoch(model, optimizer, loss_fn, num_classes, criterion, w_decay, loader, is_train=True, log_grad=False):
    tot_loss = 0.0
    tot_acc  = 0.0
    n_batches = 0
    grads_snap = None

    for batch in loader:
        batch_tf = torch_batch_to_tf(batch)
        if is_train:
            loss, acc, grads = train_step(model, optimizer, loss_fn, num_classes, criterion, w_decay, batch_tf)
            if log_grad:
                grads_snap = grads
        else:
            loss, acc = eval_step(model, loss_fn, num_classes, criterion, w_decay, batch_tf)

        tot_loss += loss
        tot_acc  += acc
        n_batches += 1

    mean_loss = tot_loss / max(1, n_batches)
    mean_acc  = tot_acc  / max(1, n_batches)
    return mean_loss, mean_acc, grads_snap


def get_args():
    parser = argparse.ArgumentParser(description="Train QGNN on graph data (TFQ)")
    parser.add_argument('--dataset', type=str, default='MUTAG', help='Dataset name (e.g., MUTAG, ENZYMES, CORA)')
    parser.add_argument('--train_size', type=float, default=0.7)
    parser.add_argument('--eval_size', type=float, default=0.5)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-2)
    parser.add_argument('--w_decay', type=float, default=0.0)
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
                        help="Which model to run")
    parser.add_argument('--graphlet_size', type=int, default=10)

    parser.add_argument('--criterion', type=str, default='crossentropy',
                        choices=['crossentropy', 'MSE', 'BCE', 'L1'],
                        help="Which loss function to train model")
    return parser.parse_args()


def main(args):
    args.node_qubit = args.graphlet_size  
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    dataset, train_loader, test_loader, task_type = load_dataset(
        name=args.dataset,
        path='../data',
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size
    )

    node_input_dim = dataset[0].x.shape[1] if dataset[0].x is not None else 1
    edge_input_dim = 0
    if dataset[0].edge_attr is not None:
        shp = dataset[0].edge_attr.shape
        edge_input_dim = shp[0] if len(shp) < 2 else shp[1]
    edge_input_dim = max(edge_input_dim, 1)

    if args.dataset.upper() == "ZINC":
        num_classes = 1
    else:
        num_classes = dataset.num_classes

    if args.task != 'graph':
        raise NotImplementedError("Node classification cho TFQ chưa được port trong file này.")

    if args.model == 'qgnn':
        model = QGNNGraphClassifierTFQ(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            graphlet_size=args.node_qubit,
            hop_neighbor=args.num_gnn_layers,
            num_classes=num_classes,
            hidden_dim=args.hidden_channels,
            dropout=0.3,
            one_hot=False,
        )
    elif args.model == 'handcraft':
        raise NotImplementedError("HandcraftGNN (TFQ) chưa được port trong file này.")
    else:
        raise NotImplementedError(f"Model '{args.model}' thuộc baseline PyTorch; chưa port TF ở đây.")

    loss_fn = make_loss(args.criterion, num_classes)
    steps_per_epoch = max(1, len(train_loader))
    optimizer, schedule = make_optimizer_and_schedule(args.lr, args.step_size, args.gamma, steps_per_epoch)
    print(f"{timestamp}\n")
    print(f"Training model {args.model} on {args.dataset} with graphlet_size={args.graphlet_size}, "
          f"epochs={args.epochs}, lr={args.lr}, step_size={args.step_size}, gamma={args.gamma}, "
          f"criterion={args.criterion}")
    best_loss = float('inf')
    patience = 10
    wait = 0
    model_save_path = os.path.join(result_dir, 'model', f"{timestamp}_{args.model}_{args.dataset.lower()}_tfq.keras")

    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    start = time.time()
    step_plot = args.epochs // 10 if args.epochs > 10 else 1

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, grads = run_epoch(
            model, optimizer, loss_fn, num_classes, args.criterion, args.w_decay, train_loader,
            is_train=True, log_grad=args.gradient
        )
        te_loss, te_acc, _ = run_epoch(
            model, optimizer, loss_fn, num_classes, args.criterion, args.w_decay, test_loader,
            is_train=False, log_grad=False
        )

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)

        if epoch % step_plot == 0:
            if num_classes == 1:
                print(f"Epoch {epoch:02d} | Train MSE: {tr_loss:.4f} | Test MSE: {te_loss:.4f}")
            else:
                print(f"Epoch {epoch:02d} | Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | "
                      f"Test Loss: {te_loss:.4f}, Acc: {te_acc:.4f}")

        if args.save_model and te_loss < best_loss:
            best_loss = te_loss
            wait = 0
            model.save_weights(model_save_path)
        else:
            wait += 1
            if wait >= patience and args.save_model:
                print(f"EarlyStopping triggered at epoch {epoch}. Best val loss: {best_loss:.4f}")
                break

        if args.gradient:
            with open(param_file, "a") as f_param:
                f_param.write("=" * 40 + f" Epoch {epoch} " + "=" * 40 + "\n")
                for v in model.trainable_variables:
                    f_param.write(f"{v.name}:\n{v.numpy()}\n\n")
            if grads is not None:
                with open(grad_file, "a") as f_grad:
                    f_grad.write("=" * 40 + f" Epoch {epoch} " + "=" * 40 + "\n")
                    for v, g in zip(model.trainable_variables, grads):
                        if g is None:
                            f_grad.write(f"{v.name}: No gradient (None)\n")
                        else:
                            f_grad.write(f"{v.name}:\n{g.numpy()}\n\n")

    end = time.time()
    print(f"Total execution time: {end - start:.6f} seconds")

    if args.plot:
        epochs_range = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.suptitle(f"{args.model.upper()}(TFQ) on {args.dataset.upper()}", fontsize=14)

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
                name=args.dataset, path='../data',
                eval_size=args.eval_size, batch_size=args.batch_size, seed=args.seed + each
            )
            ev_loss, ev_acc, _ = run_epoch(
                model, optimizer, loss_fn, num_classes, args.criterion, args.w_decay,
                eval_loader, is_train=False, log_grad=False
            )
            if num_classes == 1:
                accuracies.append(-ev_loss)
            else:
                accuracies.append(ev_acc)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1)
        if num_classes == 1:
            print(f"{args.model}(TFQ) Mean (-Loss): {mean_acc:.4f} ± {std_acc:.3f}")
        else:
            print(f"{args.model}(TFQ) Mean Accuracy: {mean_acc:.4f} ± {std_acc:.3f}")


if __name__ == "__main__":
    args = get_args()
    if args.gradient:
        header = "="*10 + f"{timestamp}_{args.model}_{args.graphlet_size}_{args.dataset.lower()}_{args.epochs}epochs_lr{args.lr}_{args.gamma}over{args.step_size}" + "="*10
        with open(param_file, "w") as f_param:
            f_param.write(header + "\n")
        with open(grad_file, "w") as f_grad:
            f_grad.write(header + "\n")
    main(args)
