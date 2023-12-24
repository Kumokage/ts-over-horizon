from sklearn.base import BaseEstimator
from typing import Tuple, Optional
from wishart import Wishart
from scipy import stats

import itertools
import numpy as np
import numpy.typing as npt
from tqdm import tqdm


class PredictiveClustering(BaseEstimator):
    def __init__(self, K: int, L: int, clustering = None,
                 distance_metric=(lambda x, y: np.linalg.norm(x-y, axis=1)), 
                 choose_prediction: str = 'mode',
                 eps: float = 5e-3, unpredicted_ratio: float = 3,
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
        self.verbose = verbose

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

    def fit(self, X: npt.NDArray, y: Optional[npt.NDArray] = None) -> BaseEstimator:
        self.generate_motives(X)
        return self

    def predict(self, X: npt.NDArray, prediction_range: int = 1) -> npt.NDArray:
        predicted_points = np.array([])
        iter_throw = (tqdm(range(prediction_range))
                      if self.verbose != 0 else range(prediction_range))
        for _ in iter_throw:
            predictions, distances = self.predict_set(
                np.append(X, predicted_points))
            prediction = self.choose_single_prediction(predictions, distances)
            predicted_points = np.append(predicted_points, np.array([prediction]))
        return predicted_points
