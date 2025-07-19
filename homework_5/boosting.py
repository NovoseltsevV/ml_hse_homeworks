from __future__ import annotations

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from typing import Optional


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        early_stopping_rounds: int | None = None,
        subsample: float = 1.0,
        bagging_temperature: float = 1.0,
        bootstrap_type: str | None = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        rsm: float = 1.0,
        quantization_type: str | None = None,
        nbins: int = 255
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.rsm = rsm
        self.quantization_type = quantization_type
        self.nbins= nbins
        self.feature_importances_ = None

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(- y * z)

    def partial_fit(self, X, y, train_pred):
        model = self.base_model_class(**self.base_model_params)

        if self.bootstrap_type == 'Bernoulli':
            n_subsample = int(self.subsample * X.shape[0])
            mask = np.random.choice(X.shape[0], n_subsample, replace=True)
            s = - self.loss_derivative(y[mask], train_pred[mask]).squeeze()
            _ = model.fit(X[mask], s)

        elif self.bootstrap_type == 'Bayesian':
            s = - self.loss_derivative(y, train_pred)
            w = np.random.uniform(0, 1, size=X.shape[0])
            weights = (- np.log(w)) ** self.bagging_temperature
            _ = model.fit(X, s, sample_weight=weights)
        
        else:
            s = - self.loss_derivative(y, train_pred)
            _ = model.fit(X, s)

        self.models.append(model)

        return model
    
    def select_features(self, X):
        n_features = int(self.rsm * X.shape[1])
        selected_features = np.random.choice(X.shape[1], n_features, replace=False)
        selected_features = np.sort(selected_features)
        return selected_features
    
    def quantize_features(self, X):
        if self.quantization_type == 'uniform':
            bins_thresholds = np.linspace(np.min(X, axis=0), np.max(X, axis=0), 
                                self.nbins, axis=-1)
            binned_X = np.array([
                np.digitize(X[:, i], bins_thresholds[i], right=True) for i in range(X.shape[1])
            ]).T
        
        elif self.quantization_type == 'quantile':
            quantiles = np.linspace(0, 1, self.nbins)
            bins_thresholds = np.quantile(X, quantiles, axis=0).T
            binned_X = np.array([
                np.digitize(X[:, i], bins_thresholds[i], right=True) for i in range(X.shape[1])
            ]).T
        
        else:
            binned_X = X
        
        return binned_X


    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        train_pred = np.zeros(y_train.shape[0])
        val_pred = np.zeros(y_val.shape[0]) if (X_val is not None) else None

        counter = 0
        best_val_loss = np.inf
        self.feature_importances_ = np.zeros(X_train.shape[1])

        for i in range(self.n_estimators):
            new_estimator = self.partial_fit(X_train, y_train, train_pred)
            new_train_pred = new_estimator.predict(X_train)
            new_gamma = self.find_optimal_gamma(y_train, train_pred, new_train_pred)
            self.gammas.append(new_gamma)
            train_pred += self.learning_rate * new_gamma * new_train_pred

            self.history['train_roc_auc'].append(roc_auc_score(y_train, self.sigmoid(train_pred)))
            self.history['train_loss'].append(self.loss_fn(y_train, train_pred))

            self.feature_importances_ += new_estimator.feature_importances_

            if X_val is not None:
                val_pred += self.learning_rate * new_gamma * new_estimator.predict(X_val)
                self.history['val_roc_auc'].append(roc_auc_score(y_val, self.sigmoid(val_pred)))
                val_loss = self.loss_fn(y_val, val_pred)
                self.history['val_loss'].append(val_loss)
                if self.early_stopping_rounds is not None:
                    if val_loss < best_val_loss:
                        counter = 0
                        best_val_loss = val_loss
                    else:
                        counter += 1
                    
                    if counter == self.early_stopping_rounds:
                        print(f'early stopping at iteration {i}')
                        break
        
        self.feature_importances_ = self.feature_importances_ / self.feature_importances_.sum()

        if plot:
            self.plot_history(X_train, y_train, name='train')
            self.plot_history(X_val, y_val, name='val')

    def predict_proba(self, X):
        logits = np.zeros(X.shape[0])
        for i in range(len(self.models)):
            pred = self.models[i].predict(X)
            logits += self.learning_rate * self.gammas[i] * pred
        probs = np.zeros((X.shape[0], 2))
        probs[:, 1] = self.sigmoid(logits)
        probs[:, 0] = 1 - probs[:, 1]
        return probs

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]
    
    def score(self, X, y):
        return score(self, X, y)
        
    def plot_history(self, X, y, name='val'):
        fig, ax = plt.subplots(ncols=2, figsize=(15, 10))
        n_arr = range(1, len(self.models) + 1)
        pred = np.zeros(X.shape[0])
        loss_history = []
        auc_roc_history = []
        for i in range(len(self.models)):
            pred += self.learning_rate * self.gammas[i] * self.models[i].predict(X)
            loss_history.append(self.loss_fn(y, pred))
            auc_roc_history.append(roc_auc_score(y, self.sigmoid(pred)))
        ax[0].plot(n_arr, loss_history)
        ax[1].plot(n_arr, auc_roc_history)
    
        ax[0].set_title('Losses plot')
        ax[0].set_xlabel('n estimators')
        ax[1].set_title('Auc roc plot')
        ax[1].set_xlabel('n estimators')
        fig.suptitle(f'Plot history {name}')
        plt.tight_layout()
        plt.show()
