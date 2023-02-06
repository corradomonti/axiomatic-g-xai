import datasets
import explainers
import measures
import mlflowutils

import mlflow
import networkx as nx
import numpy as np
import torch_geometric as tg

import logging
from collections import namedtuple
import sys
import traceback

@mlflowutils.log_arguments
def run_experiment(
        dataset,
        explainer='gnn_explainer',
        num_samples=100,
        gamma=0.,
        frac_important_feat=0.1,
        frac_important_edge=0.5,
        whitebox_type=measures.LOCAL,
        measure_edges=True,
        seed=123,
        kwargs_explainer=None,
        false_as_negatives=False,
):
    kwargs_explainer = dict() if kwargs_explainer is None else kwargs_explainer
    for key, value in kwargs_explainer.items():
        mlflow.log_param(key, value)
    for key, value in dataset.__dict__.items():
        if len(str(value)) < 10:
            mlflow.log_param(key, value)
    
    logging.info(f"Seeding everything with {seed}.")
    tg.seed_everything(seed)
    generate_explainer = getattr(explainers, 'generate_' + explainer)
    
    G, X = dataset.generate()
    logging.info(f"Generated {dataset} dataset.")
    if false_as_negatives:
        logging.info("False features will be -1.")
        X = 2. * X - 1.
    A = nx.to_numpy_array(G)
    edge_index = np.array(np.where(A))
    logging.info(f"Graph has {G.number_of_nodes()} nodes "
                 f"and {G.number_of_edges()} edges, feature matrix "
                 f"has shape {X.shape} (average: {np.mean(X)}).")
    
    num_samples = min(num_samples, G.number_of_nodes())
    nodes_to_explain = np.random.choice(G.number_of_nodes(), num_samples, replace=False)
    logging.info(f"Generated {len(nodes_to_explain)} nodes to explain%s.",
        "" if len(nodes_to_explain) > 10 else (" " + str(nodes_to_explain)))
    
    logging.info(f"Measuring quality of {explainer}({kwargs_explainer}) with {whitebox_type}...")
    explainer_model = generate_explainer(**kwargs_explainer)
    feat_quality, (global_score, local_score, weighted_score), saved_files = \
        measures.measure_quality(explainer_model,
            edge_index, X, whitebox_type=whitebox_type,
            nodes_to_explain=nodes_to_explain,
            measure_edges=measure_edges, gamma=gamma,
            frac_important_feat=frac_important_feat, frac_important_edge=frac_important_edge,
        )
    
    mlflow.log_metric('feat_quality', feat_quality)
    mlflow.log_metric('edge_quality_global', global_score)
    mlflow.log_metric('edge_quality_local', local_score)
    mlflow.log_metric('edge_quality_weighted', weighted_score)
    for saved_file in saved_files:
        mlflow.log_artifact(saved_file)
    
    logging.info("Done.")
    return feat_quality, (global_score, local_score, weighted_score)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
        
    experiment_id = "axiomatic-benchmark-42"
    mlflow.set_experiment(experiment_id)
    
    EXPLAINER_NUM_HOP = 2
    explainers_and_kwargs = (
            ('graphlime', {'hop': EXPLAINER_NUM_HOP}),
            ('integrated_gradients', {}),
            ('deconvolution', {}),
            ('lrp', {}),
            # ('gnn_explainer', {'epochs': 10, 'num_hops': EXPLAINER_NUM_HOP}),
            # ('gnn_explainer', {'epochs': 100, 'num_hops': EXPLAINER_NUM_HOP}),
            ('gnn_explainer', {'epochs': 1000, 'num_hops': EXPLAINER_NUM_HOP}),
    )
    
    Params = namedtuple(
        'Params', 'real,  pos, imp, false_as_negatives')
    synth_params = [
        Params    (False, 0.1, 0.5, False             ),
        Params    (False, 0.3, 0.5, False             ),
        Params    (False, 0.5, 0.5, False             ),
        Params    (False, 0.7, 0.5, False             ),
        Params    (False, 0.9, 0.5, False             ),
        Params    (False, 0.5, 0.1, False             ),
        Params    (False, 0.5, 0.2, False             ),
        Params    (False, 0.5, 0.3, False             ),
        Params    (False, 0.5, 0.4, False             ),
        Params    (False, 0.1, 0.5, True              ),
        Params    (False, 0.3, 0.5, True              ),
        Params    (False, 0.5, 0.5, True              ),
        Params    (False, 0.7, 0.5, True              ),
        Params    (False, 0.9, 0.5, True              ),
        Params    (False, 0.5, 0.1, True              ),
        Params    (False, 0.5, 0.2, True              ),
        Params    (False, 0.5, 0.3, True              ),
        Params    (False, 0.5, 0.4, True              ),
    ]
    real_params = [
        Params    (True,  0.5, 0.1, False             ),
        Params    (True,  0.5, 0.3, False             ),
        Params    (True,  0.5, 0.5, False             ),
    ]

    params_seed = (
        [(p, 42 + seed) for seed in range(4) for p in synth_params] +
        [(p, 666) for p in real_params]
    )

    
    for param, seed in params_seed:
        if param.real:
            dataset = datasets.FacebookDataset()
        else:
            dataset = datasets.ErdosRenyi(100, mean_degree=16, num_features=50,
                            frac_positive_feat=param.pos)
        
        for explainer, kwargs_explainer in explainers_and_kwargs:
            for whitebox_type in measures.WHITEBOX_TYPES:
                kwargs_explainer.update({'seed': seed})
                with mlflow.start_run():
                    with mlflowutils.logfile_artifact():
                        try:
                            run_experiment(
                                    dataset=dataset,
                                    explainer=explainer,
                                    gamma=1.,
                                    whitebox_type=whitebox_type,
                                    frac_important_feat=param.imp,
                                    frac_important_edge=0.5,
                                    seed=seed,
                                    kwargs_explainer=kwargs_explainer,
                                    false_as_negatives=param.false_as_negatives,
                            )
                        except Exception: # pylint: disable=broad-except
                            mlflow.set_tag('crashed', True)
                            with open("exception.txt", 'w') as _f:
                                traceback.print_exc(file=_f)
                            mlflow.log_artifact("exception.txt")
                            traceback.print_exc()
                
                mlflow.search_runs().to_csv(
                    f"../data/{experiment_id}.csv", index=False)
    print("Done ✅")
        
if __name__ == '__main__':
    main()
