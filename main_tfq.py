import os
import argparse
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from model_tfq import QGNNGraphClassifierTFQ

from data import load_dataset, eval_dataset, random_split

def torch_batch_to_tf(batch):
    x = batch.x.cpu().numpy().astype(np.float32) if batch.x is not None else None
    edge_index = batch.edge_index.cpu().numpy().astype(np.int32)
    edge_attr = batch.edge_attr.cpu().numpy().astype(np.float32) if batch.edge_attr is not None else None
    bvec = batch.batch.cpu().numpy().astype(np.int32)  # [N]
    y = batch.y.cpu().numpy()
    if y.ndim == 1:
        y = y[:, None]  

    x = tf.convert_to_tensor(x if x is not None else np.zeros((bvec.shape[0], 1), dtype=np.float32))
    edge_index = tf.convert_to_tensor(edge_index)
    edge_attr = tf.convert_to_tensor(edge_attr) if edge_attr is not None else None
    bvec = tf.convert_to_tensor(bvec)
    y = tf.convert_to_tensor(y.astype(np.float32))

    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "batch": bvec, "y": y}


def make_tf_labels(y, num_classes):
    if num_classes == 1:
        return tf.reshape(y, [-1, 1])
    else:
        y = tf.cast(tf.squeeze(y, axis=-1), tf.int32)
        return y


@tf.function
def train_step(model, optimizer, loss_fn, num_classes, batch_tf):
    with tf.GradientTape() as tape:
        logits = model(batch_tf["x"], batch_tf["edge_attr"], batch_tf["edge_index"], batch_tf["batch"], training=True)
        if num_classes == 1:
            y_true = make_tf_labels(batch_tf["y"], num_classes)
            loss = loss_fn(y_true, logits)
            acc = tf.constant(0.0)
        else:
            y_true = make_tf_labels(batch_tf["y"], num_classes) 
            loss = loss_fn(y_true, logits)
            preds = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
            acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_true), tf.float32))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, acc


@tf.function
def eval_step(model, loss_fn, num_classes, batch_tf):
    logits = model(batch_tf["x"], batch_tf["edge_attr"], batch_tf["edge_index"], batch_tf["batch"], training=False)
    if num_classes == 1:
        y_true = make_tf_labels(batch_tf["y"], num_classes)
        loss = loss_fn(y_true, logits)
        acc = tf.constant(0.0)
    else:
        y_true = make_tf_labels(batch_tf["y"], num_classes)
        loss = loss_fn(y_true, logits)
        preds = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_true), tf.float32))
    return loss, acc


def run_epoch(model, optimizer, loss_fn, num_classes, loader, is_train=True):
    tot_loss = 0.0
    tot_acc = 0.0
    n_batches = 0
    for batch in loader:
        batch_tf = torch_batch_to_tf(batch)
        if is_train:
            loss, acc = train_step(model, optimizer, loss_fn, num_classes, batch_tf)
        else:
            loss, acc = eval_step(model, loss_fn, num_classes, batch_tf)
        tot_loss += float(loss.numpy())
        tot_acc += float(acc.numpy())
        n_batches += 1
    mean_loss = tot_loss / max(1, n_batches)
    mean_acc = tot_acc / max(1, n_batches)
    return mean_loss, mean_acc

def get_args():
    parser = argparse.ArgumentParser(description="Train QGNN (TFQ) on graph data")
    parser.add_argument('--dataset', type=str, default='MUTAG')
    parser.add_argument('--train_size', type=float, default=0.7)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--eval_size', type=float, default=0.5)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)

    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--w_decay', type=float, default=0.0)

    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--graphlet_size', type=int, default=4)

    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--seed', type=int, default=1712)
    parser.add_argument('--task', type=str, default='graph', choices=['graph']) 

    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--results', action='store_true')

    return parser.parse_args()


def main(args):
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    dataset, train_loader, test_loader, task_type = load_dataset(
        name=args.dataset, path='../data',
        train_size=args.train_size, test_size=args.test_size,
        batch_size=args.batch_size
    )
    assert task_type == 'graph'

    node_input_dim = dataset[0].x.shape[1] if dataset[0].x is not None else 1
    edge_input_dim = 0
    if dataset[0].edge_attr is not None:
        shp = dataset[0].edge_attr.shape
        edge_input_dim = shp[0] if len(shp) < 2 else shp[1]
    edge_input_dim = max(edge_input_dim, 1)

    num_classes = 1 if args.dataset.upper() == "ZINC" else dataset.num_classes
    model = QGNNGraphClassifierTFQ(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        graphlet_size=args.graphlet_size,
        hop_neighbor=args.num_gnn_layers,
        num_classes=num_classes,
        hidden_dim=args.hidden_channels,
        dropout=args.dropout,
        one_hot=False,
    )
    if num_classes == 1:
        loss_fn = tf.keras.losses.MeanSquaredError()
        metrics_name = "RMSE"
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics_name = "ACC"

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"{timestamp}\n")
    print(f"Training TFQ model on {args.dataset} | graphlet_size={args.graphlet_size} | "
          f"epochs={args.epochs} | lr={args.lr}")
    best_val = np.inf
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, optimizer, loss_fn, num_classes, train_loader, is_train=True)
        te_loss, te_acc = run_epoch(model, optimizer, loss_fn, num_classes, test_loader, is_train=False)

        if num_classes == 1:
            print(f"Epoch {epoch:02d} | Train MSE: {tr_loss:.4f} | Test MSE: {te_loss:.4f}")
        else:
            print(f"Epoch {epoch:02d} | Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | "
                  f"Test Loss: {te_loss:.4f}, Acc: {te_acc:.4f}")

        if te_loss < best_val:
            best_val = te_loss
            os.makedirs("../results/model", exist_ok=True)
            model.save_weights(f"../results/model/{timestamp}_{args.dataset.lower()}_tfq.keras")

    end = time.time()
    print(f"Total execution time: {end - start:.2f}s")
    if args.results and num_classes != 1:
        accs = []
        num_runs = 30
        for k in range(num_runs):
            eval_loader = eval_dataset(name=args.dataset, path='../data',
                                       eval_size=args.eval_size, batch_size=args.batch_size, seed=args.seed + k)
            # chạy eval nhanh
            _, eval_acc = run_epoch(model, optimizer, loss_fn, num_classes, eval_loader, is_train=False)
            accs.append(eval_acc)
        print(f"TFQ Mean Acc: {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.3f}")


if __name__ == "__main__":
    args = get_args()
