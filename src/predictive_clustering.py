from sklearn.base import BaseEstimator
from typing import Callable, Tuple, Optional
from multiprocessing import Process, Pipe, cpu_count
from .wishart import Wishart
from scipy import stats

import itertools
import os
import pickle
import numpy as np
import numpy.typing as npt
import math
from tqdm import tqdm

def transform(X: npt.NDArray, pattern: npt.NDArray) -> npt.NDArray:
    N = len(X)
    indexes = np.array([0, *np.cumsum(pattern)])
    return_n = N-indexes[-1]
    index_n = len(indexes)
    indexes = np.repeat(indexes.reshape(1, index_n), return_n, axis=0)
    indexes += np.repeat(np.arange(return_n), index_n).reshape((return_n, index_n))
    return X[indexes]

class PredictiveClustering(BaseEstimator):
    def __init__(self, K: int, L: int, clustering = None,
                 distance_metric=(lambda x, y: np.linalg.norm(x-y, axis=1)), 
                 choose_prediction: str = 'mode',
                 eps: float = 5e-3, unpredicted_ratio: float = 3,
                 healing_method: str | Callable[[npt.NDArray], npt.NDArray] | None = None,
                 caching: bool = True, n_jobs: int = 1,
                 verbose: int = 0) -> None:
        """
            Parameters
            ----------
            K : int
                Range of values in pattern, mean length between observation
                in time series.
            L : int
                Length of patterns.
            clustering : sklearn.BaseEstimator
                Witch clustering algorithm use. Default without clustering. 
                For now support only Wishart clustering algorithm.
            distance_metric: Callable[[x: np.array, y: np.array], float]
                Metric for calculating distances between motives.
                Default is Euclidean norm.
            choose_prediction: str
                Method for choosing single prediction. Default is mode.
            eps: float
                Acceptable distance between motives for prediction.
            unpredicted_ratio: float
                Ratio between two largest clusters for marking point 
                not predictable. Only used when clustering approach used for 
                choosing single prediction. Default value is 3.
            healing_method: str | Callable[[npt.NDArray], npt.NDArray] | None 
                Healing approach that will be used in healing method or with 
                predict if flag is provided. Default is None, no healing.
            caching: bool
                Flag show if caching for motives should be used. Default is True. 
                Cache is saved to .motives file.
            n_jobs: int
                Number of jobs for multiprocessing. Default value is 1. If value is -1,
                use logical CPU cores number.
            verbose : int
                Control how verbose logging should be. Default without logging.
        """
        self.K = K
        self.L = L
        self.clustering = clustering
        self.distance_metric = distance_metric
        self.choose_prediction = choose_prediction
        self.eps = eps
        self.unpredicted_ratio = unpredicted_ratio
        self.healing_method = healing_method
        self.caching = caching
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()
        self.verbose = verbose

        self.motives = []
        self.generate_patterns()

    def generate_patterns(self) -> None:
        patterns = itertools.product(np.arange(1, self.K+1), repeat=self.L)
        self.patterns = np.array(list(patterns))


    @staticmethod
    def _generate_motives(patterns: npt.NDArray, X: npt.NDArray, 
                          transform, clustering=None,
                          connector = None, verbose: int = 0) -> list:
        iter_throw = (patterns
                      if verbose == 0 else tqdm(patterns))
        motives = []

        if clustering is None:
            for pattern in iter_throw:
                motives.append(transform(X, pattern))
            if connector:
                connector.send(motives)
            return motives

        for pattern in iter_throw:
            samples = transform(X, pattern)
            clustering.fit(samples)
            centers = None

            if type(clustering) is Wishart:
                cluster_object = clustering.clusters_to_objects
                for label in range(len(cluster_object)):
                    cluster_elements = samples[cluster_object[label]]

                    if label == 0 or len(cluster_elements) == 0:
                        continue

                    center = np.mean(cluster_elements, axis=0)
                    centers = (np.vstack((centers, center))
                               if centers is not None else center)

                motives.append(centers)
            else:
                raise NotImplementedError(
                    f"For now we don't support clustering by {type(clustering)}")
        if connector:
            connector.send(motives)
        return motives

    def generate_motives(self, X: npt.NDArray):
        if self.n_jobs == 1:
            self.motives = self._generate_motives(
                    self.patterns, X, transform, self.clustering, verbose=self.verbose)
        else:
            part_size = math.ceil(self.patterns.shape[0] / self.n_jobs)
            conn_to, conn_from = Pipe()
            processes = []
            iter_throw = (range(0, self.patterns.shape[0], part_size)
                          if self.verbose == 0
                          else tqdm(range(0, self.patterns.shape[0], part_size)))
            for i in iter_throw:
                processes.append(
                    Process(
                        target=PredictiveClustering._generate_motives, 
                        args=(self.patterns[i:i+part_size], X, transform, self.clustering, conn_from, 0, )
                    )
                )
            
            for process in processes:
                process.start()

            for process in processes:
                self.motives.extend(conn_to.recv())


    def predict_set(self, X: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        predictions = []
        distances = []

        for i in range(len(self.patterns)):
            pattern = -np.cumsum(self.patterns[i][::-1])[::-1]
            new_motive = X[pattern]

            current_distances = self.distance_metric(
                new_motive, self.motives[i][:, :-1])
            current_predictions = self.motives[i][:, -
                                                  1][current_distances < self.eps]
            current_distances = current_distances[current_distances < self.eps]

            distances.extend(current_distances)
            predictions.extend(current_predictions)

        return np.array(predictions), np.array(distances)

    def choose_single_prediction(self, predictions, distances):
        if len(predictions) == 0:
            return np.nan

        match self.choose_prediction:
            case "mode":
                return stats.mode(predictions, keepdims=True)[0]
            case "mean":
                return np.mean(predictions)
            case "wmean":
                weights = distances / distances.sum()
                return np.mean(weights*predictions)
            case BaseEstimator():
                self.choose_prediction.fit(predictions.reshape(-1, 1))
                labels = self.choose_prediction.labels_
                u_labels, counts = np.unique(
                    labels[labels > -1], return_counts=True)

                max_clusters = np.sort(counts)[-2:]
                if (u_labels.size > 1 and
                        max_clusters[1] / max_clusters[0] < self.unpredicted_ratio):
                    return np.nan

                if u_labels.size > 0:
                    return predictions[labels == u_labels[counts.argmax()]].mean()

    def healing(self, X: npt.NDArray, predicted_points: npt.NDArray) -> npt.NDArray:
        match self.healing_method:
            case Callable():
                return self.healing_method(np.append(X, predicted_points)[predicted_points.shape[0]:])
            case _:
                return predicted_points

    def fit(self, X: npt.NDArray, y: Optional[npt.NDArray] = None, rewrite_cache: bool = False) -> BaseEstimator:
        if self.caching:
            if os.path.isfile(".motives") and not rewrite_cache:
                with open(".motives", 'rb') as f:
                    self.motives = pickle.load(f)
            else:
                self.generate_motives(X)
                with open(".motives", 'wb') as f:
                    pickle.dump(self.motives, f)
        else:
            self.generate_motives(X)
        return self

    def predict(self, X: npt.NDArray, prediction_range: int = 1, use_healing: bool = False) -> npt.NDArray:
        predicted_points = np.array([])
        iter_throw = (tqdm(range(prediction_range))
                      if self.verbose != 0 else range(prediction_range))
        for _ in iter_throw:
            predictions, distances = self.predict_set(
                np.append(X, predicted_points))
            prediction = self.choose_single_prediction(predictions, distances)
            predicted_points = np.append(predicted_points, np.array([prediction]))

        if use_healing:
            predicted_points = self.healing(X, predicted_points)

        return predicted_points 

