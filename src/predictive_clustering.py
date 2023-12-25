from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN
from typing import Callable, Tuple, Optional
from scipy import stats

import itertools
import os
import pickle
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .utils import entropy
from .wishart import Wishart

class PredictiveClustering(BaseEstimator):
    def __init__(self, K: int, L: int, clustering = None,
                 distance_metric=(lambda x, y: np.linalg.norm(x-y, axis=1)), 
                 choose_prediction: str = 'dbscan_mode',
                 classify_point: str = 'dbscan',
                 point_classifier = None,
                 feature_extractor = None,
                 entropy_max: float = 1.9,
                 eps: float = 0.01, unpredicted_ratio: float = 0.,
                 healing_method: str | Callable[[npt.NDArray], npt.NDArray] | None = None,
                 caching: bool = True, verbose: int = 0) -> None:
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
            verbose : int
                Control how verbose logging should be. Default without logging.
        """
        self.K = K
        self.L = L
        self.clustering = clustering
        self.distance_metric = distance_metric
        self.choose_prediction = choose_prediction
        self.classify_point = classify_point
        self.point_classifier = point_classifier
        self.feature_extractor = feature_extractor
        self.eps = eps
        self.unpredicted_ratio = unpredicted_ratio
        self.healing_method = healing_method
        self.caching = caching
        self.verbose = verbose
        self.entropy_max = entropy_max

        if self.classify_point == 'dbscan' and self.point_classifier is None:
            self.point_classifier = DBSCAN(eps=0.005, min_samples=5)
        
        if self.classify_point == 'ml' and (self.point_classifier is None or self.feature_extractor is None):
            # чет напечатать, что метод сменился, так как не подали обученный МЛ классифайер
            self.classify_point = 'dbscan'
            self.point_classifier = DBSCAN(eps=0.005, min_samples=5)

        self.motives = []
        self.generate_patterns()

    def generate_patterns(self) -> None:
        patterns = itertools.product(np.arange(1, self.K+1), repeat=self.L)
        self.patterns = np.array(list(patterns))

    def transform(self, X: npt.NDArray, pattern: npt.NDArray) -> npt.NDArray:
        N = len(X)
        indexes = np.array([0, *np.cumsum(pattern)])
        return_n = N-indexes[-1]
        index_n = len(indexes)
        indexes = np.repeat(indexes.reshape(1, index_n), return_n, axis=0)
        indexes += np.repeat(np.arange(return_n),
                             index_n).reshape((return_n, index_n))
        return X[indexes]

    def generate_motives(self, X: npt.NDArray) -> None:
        iter_throw = (self.patterns
                      if self.verbose == 0 else tqdm(self.patterns))
        if self.clustering is None:
            self.motives = []
            for pattern in iter_throw:
                self.motives.append(self.transform(X, pattern))
            return

        self.motives = []
        for pattern in iter_throw:
            samples = self.transform(X, pattern)
            self.clustering.fit(samples)
            centers = None

            if type(self.clustering) is Wishart:
                cluster_object = self.clustering.clusters_to_objects
                for label in range(len(cluster_object)):
                    cluster_elements = samples[cluster_object[label]]

                    if label == 0 or len(cluster_elements) == 0:
                        continue

                    center = np.mean(cluster_elements, axis=0)
                    centers = (np.vstack((centers, center))
                               if centers is not None else center)

                self.motives.append(centers)
            else:
                raise NotImplementedError(
                    f"For now we don't support clustering by {type(self.clustering)}")

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

        self.point_classifier.fit(predictions.reshape(-1, 1))
        labels = self.point_classifier.labels_
        u_labels, counts = np.unique(
                labels[labels > -1], return_counts=True)

        max_clusters = np.sort(counts)[-2:]

        match self.classify_point:
            case "dbscan":
                if (u_labels.size > 1 and max_clusters[1] / max_clusters[0] < self.unpredicted_ratio) or u_labels.size == 0:
                    return np.nan
            
            case "entropy":
                freqs, _ = np.histogram(predictions, bins = np.linspace(0,1,50))
                probs = freqs / freqs.sum()
                H = entropy(probs)

                if H > self.entropy_max:
                    return np.nan

            case "boosting":
                features = self.feature_extractor(predictions)
                if self.point_classifier.predict(features):
                    return np.nan


        match self.choose_prediction:
            case "mode":
                return stats.mode(predictions, keepdims=True)[0]
            case "mean":
                return np.mean(predictions)
            case "wmean":
                weights = distances / distances.sum()
                return np.mean(weights*predictions)
            case "dbscan_mean":
                return predictions[labels == u_labels[counts.argmax()]].mean()
            case "dbscan_mode":
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

