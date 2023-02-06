
from captum.attr import IntegratedGradients, Deconvolution, LRP
from graphlime import GraphLIME
import numpy as np
import torch
import torch_geometric as tg
from tqdm import tqdm

def generate_captum(CaptumExplainer, seed=None, attr_kwargs=None, **kwargs):
    attr_kwargs = dict() if attr_kwargs is None else attr_kwargs
    def captum_explainer(model, X, edge_index, nodes_to_explain):
        if seed is not None:
            tg.seed_everything(seed)
        _, D = X.shape
        num_edges = edge_index.shape[1]
        edge_index_torch = torch.tensor(edge_index)
        X_torch = torch.tensor(X)
        y = model(X_torch, edge_index_torch)
        num_sample = len(nodes_to_explain)
        
        feat_imp = np.full((num_sample, D), np.nan)
        arcs_imp = np.full((num_sample, num_edges), np.nan)
        for sample_idx, node_idx in enumerate(tqdm(nodes_to_explain)):
            target = np.argmax(y[sample_idx])
            captum_model = tg.nn.to_captum(model, mask_type='node_and_edge', output_idx=node_idx)
            ig = CaptumExplainer(captum_model, **kwargs)
            edge_mask = torch.ones(num_edges, requires_grad=True)
            ig_attr_node, ig_attr_edge = ig.attribute(
                (X_torch.unsqueeze(0), edge_mask.unsqueeze(0)),
                target=target,
                additional_forward_args=(edge_index_torch, ),
                **attr_kwargs
            )
            feat_imp[sample_idx] = np.sum(np.abs(ig_attr_node.squeeze(0).detach().numpy()), axis=0)
            arcs_imp[sample_idx] = np.abs(ig_attr_edge.squeeze(0).detach().numpy())
        return feat_imp, arcs_imp
    return captum_explainer
    
def generate_integrated_gradients(seed=None, **kwargs):
    return generate_captum(IntegratedGradients,
        seed=seed, attr_kwargs={'internal_batch_size': 1}, **kwargs)

def generate_deconvolution(seed=None, **kwargs):
    return generate_captum(Deconvolution, seed=seed, **kwargs)

def generate_lrp(seed=None, **kwargs):
    import whitebox
    import captum

    captum.attr._core.lrp.SUPPORTED_LAYERS_WITH_RULES[
        whitebox.MeanFeatureLayer
    ] = captum.attr._utils.lrp_rules.EpsilonRule

    captum.attr._core.lrp.SUPPORTED_LAYERS_WITH_RULES[
        whitebox.DegreeLayer
    ] = captum.attr._utils.lrp_rules.EpsilonRule
    
    return generate_captum(LRP, seed=seed, **kwargs)

def generate_gnn_explainer(seed=None, **kwargs):
    def gnn_explain(model, X, edge_index, nodes_to_explain):
        if seed is not None:
            tg.seed_everything(seed)
        _, D = X.shape
        num_samples = len(nodes_to_explain)
        num_edges = edge_index.shape[1]
        edge_index_torch = torch.tensor(edge_index)
        X_torch = torch.tensor(X)
        gnnexplainer = tg.nn.models.GNNExplainer(model, return_type="prob", log=False, **kwargs)
        feat_imp = np.full((num_samples, D), np.nan)
        arcs_imp = np.full((num_samples, num_edges), np.nan)
        for sample_idx, node_idx in enumerate(tqdm(nodes_to_explain)):
            feat_imp[sample_idx], arcs_imp[sample_idx] = gnnexplainer.explain_node(
                int(node_idx), X_torch, edge_index_torch,
                edge_names=torch.arange(edge_index_torch.size(1))
            )
        return feat_imp, arcs_imp
    return gnn_explain

class LogModel(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
    
    def forward(self, *args, **kwargs):
        return self.original_model.forward(*args, **kwargs).log()

def generate_graphlime(seed=None, **kwargs):
    def graphlime_explain(model, X, edge_index, nodes_to_explain):
        if seed is not None:
            tg.seed_everything(seed)        
        _, D = X.shape
        num_samples = len(nodes_to_explain)
        # num_edges = edge_index.shape[1]
        edge_index_torch = torch.tensor(edge_index)
        X_torch = torch.tensor(X)
        explainer = GraphLIME(LogModel(model), **kwargs)
        feat_imp = np.full((num_samples, D), np.nan)
        for sample_idx, node_idx in enumerate(tqdm(nodes_to_explain)):
            feat_imp[sample_idx] = explainer.explain_node(int(node_idx), X_torch, edge_index_torch)
        return feat_imp, None
    return graphlime_explain
