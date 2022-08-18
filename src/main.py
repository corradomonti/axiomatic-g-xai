import datasets
import explainers
import measures
import mlflowutils

import mlflow
import networkx as nx
import numpy as np
import torch_geometric as tg

import logging
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
        beta_or_delta='delta',
        measure_edges=True,
        seed=123,
        kwargs_explainer=None,
        two_hop=False,
        noise=False,
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
    
    logging.info(f"Measuring quality of {explainer}({kwargs_explainer}) (two_hop={two_hop}, {beta_or_delta})...")
    explainer_model = generate_explainer(**kwargs_explainer)
    feat_quality, (global_score, local_score, weighted_score), saved_files = \
        measures.measure_quality(explainer_model,
            edge_index, X,
            nodes_to_explain=nodes_to_explain,
            measure_edges=measure_edges, gamma=gamma,
            beta_or_delta=beta_or_delta,
            frac_important_feat=frac_important_feat, frac_important_edge=frac_important_edge,
            two_hop=two_hop,
            noise=noise,
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
        
    experiment_id = "axiomatic-benchmark"
    mlflow.set_experiment(experiment_id)
    for frac_important_feat in (0.5, 0.1):
        for seed in range(8):
            for false_as_negatives in (False, True):
                for num_hops in (2, ):
                    for dataset in (
                        datasets.ErdosRenyi(100, mean_degree=16, num_features=50),
                        datasets.FacebookDataset(),
                        datasets.ErdosRenyi(100, mean_degree=4, num_features=50),
                    ):
                        for two_hop in (True, False):
                            for explainer, kwargs_explainer in (
                                    ('graphlime', {'hop': num_hops}),
                                    ('integrated_gradients', {}),
                                    ('deconvolution', {}),
                                    ('lrp', {}),
                                    ('gnn_explainer', {'epochs': 10, 'num_hops': num_hops}),
                                    ('gnn_explainer', {'epochs': 100, 'num_hops': num_hops}),
                                    ('gnn_explainer', {'epochs': 1000, 'num_hops': num_hops}),
                            ):
                                for beta_or_delta in ('beta', 'delta'):
                                    if two_hop and beta_or_delta == 'beta':
                                        continue
                                    for gamma in (0., 1., 10., ):
                                        kwargs_explainer.update({'seed': seed})
                                        with mlflow.start_run():
                                            with mlflowutils.logfile_artifact():
                                                try:
                                                    run_experiment(
                                                            dataset=dataset,
                                                            explainer=explainer,
                                                            gamma=gamma,
                                                            beta_or_delta=beta_or_delta,
                                                            frac_important_feat=frac_important_feat,
                                                            seed=seed,
                                                            kwargs_explainer=kwargs_explainer,
                                                            two_hop=two_hop,
                                                            false_as_negatives=false_as_negatives,
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
