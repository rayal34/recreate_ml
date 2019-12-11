from collections import Counter
import numpy as np


class TreeNode:

    def __init__(self):
        self.column = None
        self.value = None
        self.categorical = True
        self.name = None
        self.left = None
        self.right = None
        self.leaf = False
        self.classes = Counter()

    def predict_one(self, x):

        if self.leaf:
            return self.name

        col_value = x[self.column]

        if self.categorical:
            if col_value == self.value:
                return self.left.predict_one(x)
            else:
                return self.right.predict_one(x)
        else:
            if col_value < self.value:
                return self.left.predict_one(x)
            else:
                return self.right.predict_one(x)


class DecisionTree:

    def __init__(self, criterion='entropy', num_features=None):
        self.root = None
        self.feature_names = None
        self.categorical = None
        self.criterion = self._entropy if criterion == 'entropy' else self._gini
        self.num_features = num_features

    def fit(self, X, y, feature_names=None):
        if feature_names is None or len(feature_names) != X.shape[1]:
            self.feature_names = np.arange(X.shape[1])
        else:
            self.feature_names = feature_names

        is_categorical = lambda x: isinstance(x, str) or \
                                   isinstance(x, bool) or \
                                   isinstance(x, unicode)

        self.categorical = np.vectorize(is_categorical)(X[0])

        self.root = self._build_tree(X, y)

        if not self.num_features:
            self.num_features = X.shape[1]

    def _build_tree(self, X, y):

        node = TreeNode()
        idx, value, splits = self._choose_split_index(X, y)

        if idx is None or len(np.unique(y)) == 1:
            node.leaf = True
            node.classes = Counter(y)
            node.name = node.classes.most_common(1)[0][0]
        else:
            X1, y1, X2, y2 = splits
            node.column = idx
            node.name = self.feature_names[idx]
            node.value = value
            node.categorical = self.categorical[idx]

            node.left = self._build_tree(X1, y1)
            node.right = self._build_tree(X2, y2)

        return node

    def _entropy(self, y):
        entropy = 0
        len_y = len(y)
        for val in np.unique(y):
            p_val = -(y == val).sum() / len_y
            entropy += p_val * np.log(p_val)

        return entropy

    def _gini(self, y):
        gini = 1
        len_y = len(y)
        for val in np.unique(y):
            p_val = -(y == val).sum() / len_y
            gini -= p_val ** 2

        return gini

    def _information_gain(self, y, y1, y2):
        total = self.criterion(y)

        for split in (y1, y2):
            impurity = self.criterion(y)
            total -= len(split) * impurity / len(y)

        return total

    def _make_split(self, X, y, split_idx, split_val):
        split_col = X[:, split_idx]
        if isinstance(split_val, int) or isinstance(split_val, float):
            A = split_col < split_val
            B = split_col >= split_val
        else:
            A = split_col == split_val
            B = split_col != split_val
        return X[A], y[A], X[B], y[B]

    def _choose_split_idx(self, X, y):
        split_idx, split_val, splits = None, None, None
        feature_indices = np.random.choice(X.shape[1],
                                           self.num_features,
                                           replace=False)
        max_gain = 0
        X = X[:, feature_indices]

        for feature in range(X.shape[1]):
            vals = np.unique(X[:, feature])
            if len(vals) < 2:
                continue

            for val in vals:
                temp_splits = self._make_split(X, y, feature, val)
                X_left, y_left, X_right, y_right = temp_splits
                gain = self._information_gain(y, y_left, y_right)
                if gain > max_gain:
                    max_gain = gain
                    split_idx, split_val = feature, val
                    splits = temp_splits

        return split_idx, split_val, splits

    def predict(self, X):
        return np.apply_along_axis(self.root.predict_one, axis=1, arr=X)

    def __str__(self):
        '''
        Return string representation of the Decision Tree.
        '''
        return str(self.root)


class RandomForests:

    def __init__(self, n_estimators=100, num_features):
        self.n_estimators = n_estimators
        self.num_features = num_features
        self.forest = None

    def fit(self, X, y):
        self.forest = self.build_forest(X, y, self.n_estimators)


    def _build_forest(self, X, y):
        trees = []
        feature_indices = np.arange(X.shape[1])
        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(np.arange(X.shape[0]),
                                                 X.shape[0],
                                                 replace=True)
            tree = DecisionTree(num_features=self.num_features)
            tree.fit(X[bootstrap_indices], y[bootstrap_indices])
            trees.append(tree)
        return trees

    def predict(self, X):
        answers = np.array([tree.predict(X) for tree in self.forest]).T
        return np.array([Counter(row).most_common(1)[0][0] for row in answers])

