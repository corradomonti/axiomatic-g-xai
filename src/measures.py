from betagammadelta import BetaGammaDeltaNet, BetaGammaDeltaTwoHopNet

import numpy as np
import scipy.stats
from scipy.stats import binom
import sklearn.metrics
import torch

import logging
    
def measure_stability(explainer, edge_index, X):
    _N, D = X.shape
    edge_index_torch = torch.tensor(edge_index)
    X_torch = torch.tensor(X)
            
    beta = np.zeros(D)
    delta = np.zeros(D)
    
    gamma2feature_importance = dict()
    for gamma in (0., 0.1, 1., 10., 100.,):
        model = BetaGammaDeltaNet(
                        beta=torch.tensor(beta),
                        gamma=torch.tensor(gamma),
                        delta=torch.tensor(delta),
                        x=X_torch,
                        edge_index=edge_index_torch,
            )
            
        feat_imp, _arcs = explainer(model, X, edge_index)
        
        gamma2feature_importance[gamma] = feat_imp
        # average_feat_importance = np.mean(feat_imp, axis=0)
        # print(average_feat_importance)
    
    feat_imp = np.array(list(gamma2feature_importance.values()))
    feat_imp = feat_imp / np.sum(feat_imp, axis=2)[:, :, np.newaxis]
    stability = 1 - np.median(np.std(feat_imp, axis=0).flatten())
    return stability
    
def measure_quality_kendalltau(explainer, edge_index, X, gamma=0., **kwargs):
    _N, D = X.shape
    edge_index_torch = torch.tensor(edge_index)
    X_torch = torch.tensor(X)
            
    beta = np.random.randn(D)
    delta = np.zeros(D)
    
    model = BetaGammaDeltaNet(
                    beta=torch.tensor(beta),
                    gamma=torch.tensor(gamma),
                    delta=torch.tensor(delta),
                    x=X_torch,
                    edge_index=edge_index_torch,
        )
        
    feat_imp, _arcs = explainer(model, X, edge_index, **kwargs)
    average_feat_importance = np.mean(feat_imp, axis=0)
    quality = scipy.stats.kendalltau(np.abs(beta), average_feat_importance)[0]
    return quality
    
BETA = 'beta'
DELTA = 'delta'

def compute_edge_quality(edge_index, node, original_true_edge_imp, explainer_edge_imp_node, two_hop=False):
    if two_hop:
        is_predecessor = (edge_index[1] == node)
        predecessors = edge_index[0, is_predecessor]
        two_hop_edges = np.isin(edge_index[1], predecessors)
        relevant_edges = two_hop_edges | is_predecessor
        
        is_important_predecessors = (edge_index[1] == node) & original_true_edge_imp
        important_predecessors = edge_index[0, is_important_predecessors]
        two_important_hop_edges = np.isin(edge_index[1], list(important_predecessors) + [node])
        true_edge_imp = original_true_edge_imp & two_important_hop_edges
    else:
        # Only the IN-neighborhood
        relevant_edges = (edge_index[1] == node)
        true_edge_imp = original_true_edge_imp
    
    if len(np.unique(true_edge_imp[relevant_edges])) == 1:
        if not np.any(relevant_edges):
            logging.warning(f"Node {node} has no relevant edges.")
        elif not np.any(true_edge_imp & relevant_edges):
            logging.warning(f"Node {node} has no important relevant neighbors.")
        elif np.all(true_edge_imp[relevant_edges]):
            logging.warning(f"Node {node} relvant neighbors are all important.")
        return np.nan, np.nan, np.nan
        
    global_score = sklearn.metrics.roc_auc_score(true_edge_imp & relevant_edges,
                              explainer_edge_imp_node)
    local_score = sklearn.metrics.roc_auc_score(true_edge_imp[relevant_edges],
                                  explainer_edge_imp_node[relevant_edges])

    weight_nonlocal = np.sum(relevant_edges) / len(relevant_edges)
    sample_weight = relevant_edges * (1. - weight_nonlocal) + (~relevant_edges) * weight_nonlocal
    weighted_score = sklearn.metrics.roc_auc_score(
                                 true_edge_imp & relevant_edges,
                                 explainer_edge_imp_node,
                                 sample_weight=sample_weight)
                                 
    return global_score, local_score, weighted_score

def measure_quality(explainer, edge_index, X, nodes_to_explain,
        measure_edges=False, gamma=0.,
        beta_or_delta=BETA, frac_important_feat=0.1, frac_important_edge=0.5,
        two_hop=False, noise=False,
        **explainer_kwargs):
    assert beta_or_delta in (BETA, DELTA)
    
    # Preliminaries.
    _, D = X.shape
    edge_index_torch = torch.tensor(edge_index)
    X_torch = torch.tensor(X)
    
    # Set up true feature importance.
    if ( # Probability that true_feat_imp will have all ones or all zeroes.
        (1 - (binom.pmf(0, D, frac_important_feat))) *
        (1 - (binom.pmf(D, D, frac_important_feat))) < 0.05
    ):
        raise Exception(f"Too unlikely to generate important features with frac_important_feat={frac_important_feat} and D={D}")
    
    true_feat_imp = None
    while true_feat_imp is None or all(true_feat_imp) or not any(true_feat_imp):
        true_feat_imp = (np.random.random(D) < frac_important_feat).astype(np.float64)
    
    logging.info("Features are on average %.2f important.", np.mean(true_feat_imp))
    
    beta = true_feat_imp if beta_or_delta == BETA else np.zeros(D)
    delta = true_feat_imp if beta_or_delta == DELTA else np.zeros(D)
    
    # Set up true edge importance.
    if measure_edges:
        true_edge_imp = np.random.random(size=(edge_index.shape[1], )) < frac_important_edge
        logging.info('Generated %s important edges out of %s (should be %.3f).',
            np.sum(true_edge_imp), len(true_edge_imp), frac_important_edge)
    else:
        true_edge_imp = None
        logging.info("Not measuring edge importance.")
        
    assert measure_edges is not None or any(true_edge_imp)
    
    if measure_edges and beta_or_delta == BETA and gamma == 0.:
        logging.warning("Measuring edge quality but model does not use edges.")
    
    # Setting up the white-box model and running the explainer on it.
    Whitebox = BetaGammaDeltaTwoHopNet if two_hop else BetaGammaDeltaNet
    model = Whitebox(
                    beta=torch.tensor(beta),
                    gamma=torch.tensor(gamma),
                    delta=torch.tensor(delta),
                    edge_imp=(torch.tensor(true_edge_imp) if true_edge_imp is not None else None),
                    x=X_torch,
                    edge_index=edge_index_torch,
                    noise=noise,
        )
        
    explainer_feat_imp, explainer_edge_imp = explainer(model, X, edge_index, nodes_to_explain, 
                                                        **explainer_kwargs)
    
    # Measuring results.
    quality_feat_per_node = np.array([
        sklearn.metrics.roc_auc_score(true_feat_imp, explainer_feat_imp_node)
        for explainer_feat_imp_node in explainer_feat_imp
    ])
    np.savetxt('quality_feat_per_node.txt', quality_feat_per_node, '%f')
    saved_files = ['quality_feat_per_node.txt']
    quality_feat = np.mean(quality_feat_per_node)
    
    if measure_edges and explainer_edge_imp is not None:
        quality_edge_per_node = np.array([
            compute_edge_quality(edge_index, node, true_edge_imp, explainer_edge_imp_node,
                                 two_hop=two_hop)
            for explainer_edge_imp_node, node in zip(explainer_edge_imp, nodes_to_explain)
        ])
        np.savetxt('quality_edge_per_node.txt', quality_edge_per_node, '%f')
        saved_files += ['quality_edge_per_node.txt']
        global_score, local_score, weighted_score = np.nanmean(quality_edge_per_node, axis=0)
    else:
        global_score, local_score, weighted_score = np.nan, np.nan, np.nan
        
    logging.info("quality of features %s, weighted quality of edges %s", 
                 quality_feat, weighted_score)
    
    logging.info("global quality of edges %s, local quality of edges %s", 
                 global_score, local_score)
    
    return quality_feat, (global_score, local_score, weighted_score), saved_files
