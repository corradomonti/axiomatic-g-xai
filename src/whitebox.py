
import numpy as np
import torch
import torch_geometric as tg

class MaskedMessagePassing(tg.nn.MessagePassing):
    """ Subclass of MessagePassing that allows to apply a mask on the edges. """
    
    def __init__(self, edge_imp=None, **kwargs):
        super().__init__(**kwargs)
        self.edge_imp = edge_imp
        
    def message_unmasked(self, x_j):
        """ Subclasses should use this method as if it was `message`.
        """
        return x_j
    
    def message(self, x_j, edge_names=None):
        """ `edge_names` should be a 1D Tensor containing edge indexes relative to edge_imp:
            e.g., if it contains 1, 3, 5 the corresponding elements of edge_imp will be considered.
            It is enough to pass to an explainer `edge_names=torch.arange(edge_index.size(1))``.
        """
        feat_neighbors = self.message_unmasked(x_j)
        
        if self.edge_imp is None:
            return feat_neighbors # No edge mask was given.
        
        if edge_names is None:
            if (x_j.shape[0], ) == self.edge_imp.shape:
                # Graph is the original graph, no reindex.
                return torch.einsum('e,ef->ef', self.edge_imp, feat_neighbors)
            else:
                raise Exception("edge_names not given and graph does not match with edge_imp"
                                f" ({x_j.shape} != {self.edge_imp.shape})")
        
        return torch.einsum('e,ef->ef', self.edge_imp[edge_names], feat_neighbors)


class MeanFeatureLayer(MaskedMessagePassing):
    def __init__(self, edge_imp=None):
        super().__init__(edge_imp=edge_imp, aggr="sum", flow="source_to_target", node_dim=0)
    
    def build_true_edge_index(self, edge_index, edge_names):
        if self.edge_imp is None: # No edge mask was given.
            return edge_index
        
        if edge_names is None:
            if (edge_index.shape[1], ) == self.edge_imp.shape:
                # Graph is the original graph, no reindex.
                return edge_index[:, self.edge_imp]
            else:
                raise Exception("edge_names not given and graph does not match with edge_imp"
                                f" ({edge_index.shape} != {self.edge_imp.shape})")
        
        return edge_index[:, self.edge_imp[edge_names]]
        
    def forward(self, x, edge_index, edge_names=None, **kwargs):
        N, _ = x.shape
        sum_features = self.propagate(edge_index=edge_index, x=x,
            edge_names=edge_names, **kwargs)
        true_edge_index = self.build_true_edge_index(edge_index, edge_names)
        masked_indeg = torch.bincount(true_edge_index[1, :], minlength=N)
        return sum_features / torch.maximum(masked_indeg, torch.tensor(1)).unsqueeze(1)
    
    def update(self, aggr_out, x):
        return aggr_out
    
    def message_unmasked(self, x_j):
        return x_j


class DegreeLayer(MaskedMessagePassing):
    def __init__(self, gamma, avg_degree, edge_imp=None):
        super().__init__(edge_imp=edge_imp, aggr="add", flow="source_to_target", node_dim=0)
        # self.gamma = torch.nn.parameter.Parameter(gamma)
        self.gamma = gamma
        self.avg_degree = avg_degree
        
    def forward(self, x, edge_index, **kwargs):
        return self.propagate(edge_index=edge_index, x=x, **kwargs)
    
    def message_unmasked(self, x_j):
        return (self.gamma / self.avg_degree) * x_j

class WhiteboxClassifier(torch.nn.Module):
    r"""
    Model that assigns binary class labels to nodes based on their features and on three fixed parameter beta, gamma, and delta. The model is defined by:
    $$
    \textbf{y}_i = \sigma(
        \beta^\top \textbf{x}_i + \gamma | N-(i)| + 
            \delta^\top \frac{1}{|N(i)|} \sum_{j \in N-(i)} \textbf{x}_j
    )
    $$
    where $\sigma$ is a sigmoid centered in y_mean, N is the in-neighborhood, and x are features.
    """
    
    def __init__(self, beta, gamma, neighborhood=False, edge_imp=None, x=None,
        edge_index=None):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.neighborhood = neighborhood
        avg_degree = np.bincount(edge_index[0]).mean()
        self.message_passing_layer = MeanFeatureLayer(edge_imp)
        self.degree_layer = DegreeLayer(self.gamma, avg_degree, edge_imp)
        
        # Calibrate the sigmoid, i.e. we'll use the z-scores of logits.
        y_hat = self.logits(x=x, edge_index=edge_index)
        self.y_mean = y_hat.mean()
        self.steepness = 1. / y_hat.std()

        
    def logits(self, x, edge_index, **kwargs):
        if self.neighborhood:
            feature_comp = torch.einsum('d,id->i', self.beta,
                self.message_passing_layer(x, edge_index=edge_index, **kwargs)).unsqueeze(-1)
        else:
            feature_comp = torch.einsum('d,id->i', self.beta, x).unsqueeze(-1)
        
        degree_input = (x.sum(axis=1)*0+1).unsqueeze(-1) # fix to have a gradient
        degree_comp = self.degree_layer(degree_input, edge_index=edge_index, **kwargs)
        assert feature_comp.shape == degree_comp.shape, (
            f"{feature_comp.shape}, {degree_comp.shape}"
        )
        return feature_comp + degree_comp

    def forward(self, x, edge_index, **kwargs):
        """
        Computes the class labels for the given edges and features.
        
        Args:
            x (Tensor): a matrix where x[i, j] is the j-th feature of node i.
            edge_index (Tensor): edges as an adjacency list (shape: 2 x number of edges).
        """
        y_hat = self.logits(x=x, edge_index=edge_index, **kwargs)
        assert y_hat.shape == (x.shape[0], 1)
        y = (self.steepness * (torch.squeeze(y_hat) - self.y_mean)).sigmoid()
        y_one_hot = torch.vstack([(1 - y), y]).T
        return y_one_hot

class TwoHopClassifier(WhiteboxClassifier):
    def logits(self, x, edge_index, **kwargs):
        feature_hid = self.message_passing_layer(x, edge_index=edge_index, **kwargs)
        feature_final = self.message_passing_layer(feature_hid, edge_index=edge_index, **kwargs)
        feature_comp = torch.einsum('d,id->i', self.beta,
            feature_final).unsqueeze(-1)
        
        #fix to have a gradient
        degree_input = (x.sum(axis=1)*0+1).unsqueeze(-1)
        gamma_hid = self.degree_layer(degree_input, edge_index=edge_index, **kwargs)
        degree_comp = self.degree_layer(gamma_hid, edge_index=edge_index, **kwargs)
        
        assert feature_comp.shape == degree_comp.shape, (
            f"{feature_comp.shape}, {degree_comp.shape}"
        )
        return degree_comp + feature_comp
    
