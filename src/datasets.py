import numpy as np
import networkx as nx

class DatasetLoader:
    def generate(self):
        raise NotImplementedError()

class SyntheticDataset(DatasetLoader):
    def __init__(self, num_nodes, mean_degree, num_features):
        self.num_nodes = num_nodes
        self.mean_degree = mean_degree
        self.num_features = num_features
    
    def generate_features(self):
        return np.array(
            np.random.random(size=(self.num_nodes, self.num_features)) > 0.5, np.float64)
    
    def __str__(self):
        return type(self).__name__ + \
            f"-{self.num_nodes}-nodes-{self.num_features}-features-{self.mean_degree}-degree"

class ErdosRenyi(SyntheticDataset):
    def generate(self):
        prob_link = self.mean_degree / self.num_nodes
        G = nx.erdos_renyi_graph(self.num_nodes, prob_link)
        return G, self.generate_features()
        
class BarabasiAlbert(SyntheticDataset):
    def generate(self):
        G = nx.barabasi_albert_graph(self.num_nodes, self.mean_degree // 2)
        return G, self.generate_features()

class FacebookDataset(DatasetLoader):
    def __init__(self, dataset_id='1684'):
        self.dataset_id = dataset_id
        # Loading from disk
        X_original = np.loadtxt(f"../data/raw/facebook/{dataset_id}.feat")
        edges_original = np.loadtxt(f"../data/raw/facebook/{dataset_id}.edges", dtype=np.int64)
        assert set(edges_original.flatten()) < set(map(int, X_original[:, 0]))
        
        # Discarding singletons as in https://arxiv.org/pdf/2201.11596.pdf
        nodes_with_edges = np.array(list(set(edges_original.flatten())))
        original2new = {original_id: new_id
                        for new_id, original_id in enumerate(nodes_with_edges)}
        
        # Reindexing feature matrix: original ids are not in (0, N) + we only use those with edges
        X_reindexed = X_original[
            np.array([np.where(X_original[:, 0] == i)[0] for i in nodes_with_edges]).squeeze()
        ]
        assert np.all(X_reindexed[:, 0] == nodes_with_edges)
        self.X = X_reindexed[:, 1:].astype(np.float64)
        
        # Building graph with reindexed edges
        edges = np.vectorize(original2new.__getitem__)(edges_original)
        self.G = nx.Graph()
        self.G.add_nodes_from(original2new.values())
        self.G.add_edges_from(edges)
        assert self.G.number_of_nodes() == self.X.shape[0]
    
    def generate(self):
        return self.G, self.X
        
    def __str__(self):
        return type(self).__name__ + "-" + self.dataset_id
