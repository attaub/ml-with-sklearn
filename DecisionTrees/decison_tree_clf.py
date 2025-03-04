import numpy as np
# from tree_node import TreeNode

class TreeNode:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


# Decision Tree Classifier
class DTClassifier:
    def __init__(self, max_depth=5, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        return self._predict_one(
            x,
            node.left
            if x[node.feature_index] <= node.threshold
            else node.right,
        )

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return TreeNode(value=self._most_common_label(y))

        feature_index, threshold = self._best_split(X, y)
        if feature_index is None:
            return TreeNode(value=self._most_common_label(y))

        left_mask = X[:, feature_index] <= threshold
        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return TreeNode(
            feature_index=feature_index,
            threshold=threshold,
            left=left_node,
            right=right_node,
        )

    def _best_split(self, X, y):
        best_feature_index = None
        best_threshold = None
        best_impurity = float('inf')
        impurity_func = self.gini if self.criterion == "gini" else self.entropy

        n_samples, n_features = X.shape
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                left_y, right_y = y[left_mask], y[~left_mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                weighted_impurity = (len(left_y) / n_samples) * impurity_func(
                    left_y
                ) + (len(right_y) / n_samples) * impurity_func(right_y)

                if weighted_impurity < best_impurity:
                    best_impurity = weighted_impurity
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities**2)

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def _most_common_label(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]

    def prune(self, X_val, y_val):
        self._prune_node(self.root, X_val, y_val)

    def _prune_node(self, node, X_val, y_val):
        if node is None or node.value is not None:
            return

        self._prune_node(node.left, X_val, y_val)
        self._prune_node(node.right, X_val, y_val)

        current_accuracy = np.mean(self._predict_subtree(node, X_val) == y_val)
        leaf_accuracy = np.mean(self._predict_leaf(node, X_val) == y_val)

        if leaf_accuracy >= current_accuracy:
            node.left = node.right = None
            node.value = self._most_common_label(y_val)

    def _predict_subtree(self, node, X):
        return np.array([self._predict_one(x, node) for x in X])

    def _predict_leaf(self, node, X):
        return np.full(
            X.shape[0], self._most_common_label(self._predict_subtree(node, X))
        )



from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data[:, 2:]
y = iris.target

my_clf = DTClassifier(max_depth=3)
my_clf.fit(X,y)
my_clf.predict(X[1:9])
# how to do prection
