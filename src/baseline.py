import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, GATConv, SAGEConv, TransformerConv, global_add_pool, global_mean_pool

class GIN_Node(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            mlp = MLP(in_channels if i==0 else hidden_channels,
                       hidden_channels, hidden_channels)
            self.convs.append(GINConv(nn=mlp, train_eps=False))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.dropout(x)
        return self.classifier(x)
    


class GCN_Node(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch  = in_channels if i==0 else hidden_channels
            out_ch = out_channels  if i==num_layers-1 else hidden_channels
            self.convs.append(GCNConv(in_ch, out_ch))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs[:-1]:
            x = F.sigmoid(conv(x, edge_index))
        x = self.dropout(x)
        # last layer, no activation
        return self.convs[-1](x, edge_index)


class GAT_Node(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, heads=8, dropout=0.6):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i==0 else hidden_channels * heads
            if i < num_layers-1:
                self.convs.append(
                    GATConv(in_ch, hidden_channels, heads=heads, dropout=dropout)
                )
            else:
                # final layer: single head, no concat
                self.convs.append(
                    GATConv(in_ch, out_channels, heads=1, concat=False, dropout=dropout)
                )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_attr, edge_index, batch=None):
        x = self.dropout(x)
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
        x = self.dropout(x)
        return self.convs[-1](x, edge_index)
    
class GraphSAGE_Node(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        # tầng đầu
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        # các tầng ẩn
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        # tầng đầu ra
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_attr,edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

## NOTE: Graph Task
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super().__init__()
        layers = []
        # input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        # hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class GIN_Graph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            mlp = MLP(in_dim,
                       hidden_channels, hidden_channels)
            self.convs.append(GINConv(nn=mlp, train_eps=False))
        self.dropout = nn.Dropout(0.5)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        return F.log_softmax(self.classifier(x), dim=-1)
    
class GCN_Graph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNConv(in_dim,hidden_channels))
        # self.conv1 = GCNConv(18, 64, 6)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = global_add_pool(x, batch)
        x = self.classifier(x)
        return F.log_softmax(x, dim = 1)
    
class GAT_Graph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=1):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels * heads
            self.convs.append(GATConv(in_dim, hidden_channels, heads=heads))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = global_mean_pool(x, batch)#global_add_pool(x, batch)
        return self.classifier(x)


class GraphSAGE_Graph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_dim, hidden_channels))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = global_add_pool(x, batch)
        return self.classifier(x)
    
    
class Transformer_Graph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=1):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels * heads
            self.convs.append(TransformerConv(in_dim, hidden_channels, heads=heads))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_attr, edge_index, batch=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        x = global_add_pool(x, batch)
        return self.classifier(x)