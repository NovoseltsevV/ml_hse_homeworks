import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    feature_vector_sort = np.unique(np.sort(feature_vector))
    thresholds = ((np.r_[feature_vector_sort, 0] + np.r_[0, feature_vector_sort])/2)[1:-1]
    
    left_mask = feature_vector[:, None] <= thresholds
    right_mask = ~left_mask

    p0_left = np.sum((target_vector[:, None] == 0) & left_mask, axis=0) / np.sum(left_mask, axis=0)
    p1_left = np.sum((target_vector[:, None] == 1) & left_mask, axis=0) / np.sum(left_mask, axis=0)
    p0_right = np.sum((target_vector[:, None] == 0) & right_mask, axis=0) / np.sum(right_mask, axis=0)
    p1_right = np.sum((target_vector[:, None] == 1) & right_mask, axis=0) / np.sum(right_mask, axis=0)

    n_left = np.sum(left_mask, axis=0)
    n_right = np.sum(right_mask, axis=0)

    gini_left = 1 - p0_left**2 - p1_left**2
    gini_right = 1 - p0_right**2 - p1_right**2
    ginis = - (n_left * gini_left + n_right * gini_right) / len(target_vector)

    gini_best = max(ginis)
    best_indices = np.where(ginis == gini_best)[0]
    threshold_best = thresholds[best_indices[0]]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=1, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if "depth" not in node:
            node["depth"] = 0

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if (len(sub_y) < self._min_samples_split) or (self._max_depth is not None and node["depth"] == self._max_depth):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            cur_split = feature_vector < threshold
            if (np.sum(cur_split) < self._min_samples_leaf) or (np.sum(np.logical_not(cur_split)) < self._min_samples_leaf):
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {"depth": node["depth"] + 1}, {"depth": node["depth"] + 1}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature_best = node["feature_split"]
        if self._feature_types[feature_best] == "real":
            threshold_best = node["threshold"]
            to_left = (x[feature_best] < threshold_best)
        elif self._feature_types[feature_best] == "categorical":
            threshold_best = node["categories_split"]
            to_left = (x[feature_best] in threshold_best)

        if to_left:
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)


class LinearRegressionTree(DecisionTree):
    def __init__(self, feature_types, 
                 max_depth=None, 
                 min_samples_split=1, 
                 min_samples_leaf=1,
                 quantiles=10):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._quantiles = np.linspace(0, 1, quantiles + 2)[1:-1]

    def _fit_node(self, sub_X, sub_y, node):
        if "depth" not in node:
            node["depth"] = 0
        
        if (len(sub_y) < self._min_samples_split) or (self._max_depth is not None and node["depth"] == self._max_depth):
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)
            return

        feature_best, threshold_best = None, None
        best_loss, best_split = np.inf, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]

                for threshold in np.quantile(feature_vector, self._quantiles):
                    split = feature_vector < threshold
                    if (np.sum(split) < self._min_samples_leaf) or (np.sum(~split) < self._min_samples_leaf):
                        continue
                    
                    left_model = LinearRegression().fit(sub_X[split], sub_y[split])
                    right_model = LinearRegression().fit(sub_X[~split], sub_y[~split])

                    left_loss = mean_squared_error(sub_y[split], left_model.predict(sub_X[split]))
                    right_loss = mean_squared_error(sub_y[~split], right_model.predict(sub_X[~split]))

                    loss = (np.sum(split) * left_loss + np.sum(~split) * right_loss) / len(sub_y)

                    if loss < best_loss:
                        best_loss = loss
                        best_split = split
                        feature_best = feature
                        threshold_best = threshold
            
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                ratio = {}
                for key, current_count in counts.items():
                    ratio[key] = np.sum(sub_y[sub_X[:, feature] == key]) / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))

                if len(np.unique(feature_vector)) == 1:
                    continue

                feature_vector_sort = np.unique(np.sort(feature_vector))
                thresholds = ((np.r_[feature_vector_sort, 0] + np.r_[0, feature_vector_sort])/2)[1:-1]

                for threshold in thresholds:
                    split = feature_vector < threshold
                    if (np.sum(split) < self._min_samples_leaf) or (np.sum(~split) < self._min_samples_leaf):
                        continue
                    
                    left_model = LinearRegression().fit(sub_X[split], sub_y[split])
                    right_model = LinearRegression().fit(sub_X[~split], sub_y[~split])

                    left_loss = mean_squared_error(sub_y[split], left_model.predict(sub_X[split]))
                    right_loss = mean_squared_error(sub_y[~split], right_model.predict(sub_X[~split]))

                    loss = (np.sum(split) * left_loss + np.sum(~split) * right_loss) / len(sub_y)

                    if loss < best_loss:
                        best_loss = loss
                        best_split = split
                        feature_best = feature
                        threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
            else:
                raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {"depth": node["depth"] + 1}, {"depth": node["depth"] + 1}
        self._fit_node(sub_X[best_split], sub_y[best_split], node["left_child"])
        self._fit_node(sub_X[~best_split], sub_y[~best_split], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["model"].predict(x.reshape(1, -1))[0]
        
        feature_best = node["feature_split"]
        if self._feature_types[feature_best] == "real":
            threshold_best = node["threshold"]
            to_left = (x[feature_best] < threshold_best)
        elif self._feature_types[feature_best] == "categorical":
            threshold_best = node["categories_split"]
            to_left = (x[feature_best] in threshold_best)

        if to_left:
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])
