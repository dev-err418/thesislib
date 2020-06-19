import numpy as np
import scipy.sparse as sparse
from scipy.sparse import sputils
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from copy import deepcopy


class RFParams(object):
    n_estimators = 20
    criterion = 'gini'
    max_depth = 380
    min_samples_split = 2
    min_samples_leaf = 2
    max_leaf_nodes = None
    min_impurity_decrease = 0.0
    max_features = 'log2'


class ThesisCategoricalNB(CategoricalNB):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None, skip_zero=False, encoding_map=None):
        super(ThesisCategoricalNB, self).__init__(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)
        self.encoder = OrdinalEncoder(dtype=np.uint8)
        self.skip_zero = skip_zero
        self.default_categories = []
        self.encoding_map = encoding_map

    def select_closet_by_distance(self, categories, targets, enc_map):
        _targets = list(map(lambda item: enc_map.get(item), targets))

        def calc_distance(cat1, cat2):
            cat_1 = cat1.split(";")

            if cat2 is None:
                return len(cat_1)

            cat_2 = cat2.split(";")

            dist = 0
            for idx in range(len(cat_1)):
                if cat_1[idx] != cat_2[idx]:
                    dist += 1
            return dist

        distances = np.zeros((len(categories), len(targets)))
        for idx, cat1 in enumerate(categories):
            _cat1 = enc_map.get(cat1)
            for jdx, _cat2 in enumerate(_targets):
                distances[idx, jdx] = calc_distance(_cat1, _cat2)

        return np.array(targets)[np.argmin(distances, axis=1)]

    def handle_transformation(self, X):
        if sputils.isdense(X):
            X_transformed = X
        else:
            X_transformed = X.toarray()

        shape = X_transformed.shape
        if len(shape) == 1:
            num_features = 1
        else:
            num_features = shape[1]

        if self.encoding_map is None:
            for idx in range(num_features):
                unique, counts = np.unique(X_transformed[:, idx], return_counts=True)
                if self.skip_zero:
                    maxidx = np.argmax(counts[1:])
                else:
                    maxidx = np.argmax(counts[1:])

                self.default_categories.append(unique[maxidx])

        X_transformed = self.encoder.fit_transform(X_transformed)

        return X_transformed

    def fit(self, X, y, sample_weight=None):
        X_transformed = self.handle_transformation(X)
        return super().fit(X_transformed, y, sample_weight)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        raise NotImplementedError("Partial fit is not implemented for this Categorical NB")

    def _joint_log_likelihood(self, X):
        X_transformed = deepcopy(X)
        for idx in range(self.n_features_):
            is_present = X_transformed[:, idx].reshape((-1, 1)) == self.encoder.categories_[idx]
            summed = np.sum(is_present, axis=1)
            indices = np.where(summed == 0)[0]
            if len(indices) > 0:
                if self.encoding_map is None:
                    X_transformed[indices, idx] = self.default_categories[idx]
                else:
                    X_transformed[indices, idx] = self.select_closet_by_distance(
                        X_transformed[indices, idx],
                        self.encoder.categories_[idx],
                        self.encoding_map[idx]
                    )
        X_transformed = self.encoder.transform(X_transformed)
        return super()._joint_log_likelihood(X_transformed)


class ThesisNaiveBayes(BaseEstimator, ClassifierMixin):
    """
    The idea for this class is quite straightforward
    With Naive Bayes, we make a strong conditional independence assumption on the data that we do have
    This means the given a sample X with a set of variables {x0, x1, ...,xn} the probability that the
    sample X belongs to a class c i.e P(c | X) can be expressed as :
    P(c|X) = P(c|x0) * P(c|x1) * ... * P(c|xn)
    This allows for an individual estimation of P(c|x_i)

    However when X is a mixture of types of variables, e.g continous, categorical (with multiple categories), binomial, etc
    we need to combined the results as shown in the product above.

    This classes provides an interface for combinning different naive bayes classifiers operating on different type of variables
    """
    def __init__(self, classifier_map, classes=None):
        """
        The classifier map is in this format:
        classifier_map = [[clf, [list_of_keys_for_same_variable_type]], ...]

        So assuming the data is made up of two  gaussian variables say age and height, and two categorical variables say gender and race
        the classifier_map would be something like:
        classifier_map = [[clf_gauss, ['age', 'height']], [clf_cat, ['gender', 'race']]]
        where clf_gauss = sklearn.naive_bayes.GaussianNB() and clf_cat = sklearn.naive_bayes.MultinomialNB()
        :param classifier_map:
        """
        self.classifier_map = classifier_map
        self.fitted = False
        self.is_partial = False
        self.classes_ = None
        self.labelbin = None

        if classes is not None:
            self.fit_classes(classes)

    def fit_classes(self, classes):
        self.labelbin = LabelBinarizer()
        self.labelbin.fit(classes)
        self.classes_ = self.labelbin.classes_

    def _joint_log_likelihood(self, X):
        """
        Computes the join log likelihood for the different classifiers
        :param X:
        :return:
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted.")

        _probs = np.zeros((X.shape[0], self.classes_.shape[0]))
        for idx in range(len(self.classifier_map)):
            clf, keys = self.classifier_map[idx]
            _df = X[keys]
            _probs += clf._joint_log_likelihood(_df)

        return _probs

    def fit(self, X, y):
        """
        Fits the different classifiers on the relevant part of the data and stores the
        fitted classifiers for further use
        :param X:
        :param y:
        :return:
        """
        if self.classes_ is None:
            classes = np.unique(y)
            self.fit_classes(classes)

        _y = self.labelbin.fit_transform(y)
        for idx in range(len(self.classifier_map)):
            clf, keys = self.classifier_map[idx]
            _df = X[keys]
            clf.fit(_df, y)
            self.classifier_map[idx][0] = clf

        self.fitted = True

    def partial_fit(self, X, y):
        _y = self.labelbin.fit_transform(y)

        for idx in range(len(self.classifier_map)):
            clf, keys = self.classifier_map[idx]
            _df = X[keys]
            clf.partial_fit(_df, y, self.classes_)
            self.classifier_map[idx][0] = clf

        self.fitted = True
        self.is_partial = True

    def predict_log_proba(self, X):
        if not self.fitted:
            raise ValueError("Model has not been fitted.")

        _probs = self._joint_log_likelihood(X)
        # normalize (see naive_bayes.py in sklearn for explanation!!)
        _log_prob_x = logsumexp(_probs, axis=1)
        return _probs - np.atleast_2d(_log_prob_x).T

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model has not been fitted.")
        jll = self._joint_log_likelihood(X)
        if type(jll) != "numpy.ndarray":
            jll = jll.values
        return self.classes_[np.argmax(jll, axis=1)]

    def serialize(self):
        return {
            "fitted": self.fitted,
            "classes": self.classes_,
            "classifier_map": self.classifier_map,
            "is_partial": self.is_partial
        }

    @staticmethod
    def load(serialized):
        if type(serialized) is not dict:
            raise ValueError("Serialized model has to be a dict")

        fitted = serialized.get('fitted', None)
        classes = serialized.get('classes', None)
        classifier_map = serialized.get('classifier_map', None)
        is_partial = serialized.get('is_partial', None)

        if not fitted or not classifier_map or not classes or not is_partial:
            raise ValueError("Missing required serialization entities")

        clf = ThesisNaiveBayes(classifier_map, classes)
        clf.fitted = fitted
        clf.is_partial = is_partial

        return clf


class ThesisAIMEDSymptomSparseMaker(BaseEstimator):
    def __init__(self, num_symptoms, categorical_indices=None):
        self.num_symptoms = num_symptoms
        self.categorical_indices = categorical_indices

    def fit_transform(self, df, y=None):
        symptoms = df.SYMPTOMS
        df = df.drop(columns=['SYMPTOMS'])

        dense_matrix = sparse.coo_matrix(df.values)
        symptoms = symptoms.apply(lambda v: [idx for idx in v.split(",")])

        columns = []
        rows = []
        data = []
        for idx, val in enumerate(symptoms):
            rows += [idx] * len(val)
            cols_data = []
            cols = []
            for item in val:
                _ = item.split("|")
                cols.append(int(_[0]))
                cols_data.append(int(_[1]))
            columns += cols
            data += cols_data

        symptoms_coo = sparse.coo_matrix((data, (rows, columns)), shape=(df.shape[0],self.num_symptoms))

        data_coo = sparse.hstack([dense_matrix, symptoms_coo])
        data_csc = data_coo.tocsc()

        return data_csc


class ThesisSymptomSparseMaker(BaseEstimator):
    def __init__(self, num_symptoms):
        self.num_symptoms = num_symptoms

    def fit_transform(self, df, y=None):
        symptoms = df.SYMPTOMS
        df = df.drop(columns=['SYMPTOMS'])

        dense_matrix = sparse.coo_matrix(df.values)
        symptoms = symptoms.apply(lambda v: [int(idx) for idx in v.split(",")])

        columns = []
        rows = []
        for idx, val in enumerate(symptoms):
            rows += [idx] * len(val)
            columns += val

        data = np.ones(len(rows))

        symptoms_coo = sparse.coo_matrix((data, (rows, columns)), shape=(df.shape[0],self.num_symptoms))

        data_coo = sparse.hstack([dense_matrix, symptoms_coo])

        return data_coo.tocsc()


class ThesisSparseNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier_map, classes=None):
        self.fitted = False
        self.classes = classes
        self.labelbin = None
        self.classifier_map = classifier_map

    def fit_classes(self, classes):
        self.labelbin = LabelBinarizer()
        self.labelbin.fit(classes)
        self.classes = self.labelbin.classes_

    @staticmethod
    def _get_data(keys, X):
        index, is_sparse = keys
        if not sputils.issequence(index):
            data = X[:, index]
            data = data.reshape((-1, 1))
        else:
            start, end = index
            if end is None:
                data = X[:, start:]
            else:
                data = X[:, start:end]

        if not is_sparse:
            return data.toarray()

        return data

    def fit(self, X, y):
        """
        Fits the different classifiers on the relevant part of the data and stores the
        fitted classifiers for further use
        :param X:
        :param y:
        :return:
        """
        if self.classes is None or self.labelbin is None:
            classes = self.classes if self.classes is not None else np.unique(y)
            self.fit_classes(classes)

        _y = self.labelbin.fit_transform(y)
        for idx in range(len(self.classifier_map)):
            clf, keys = self.classifier_map[idx]
            data = self._get_data(keys, X)
            clf.fit(data, y)

            self.classifier_map[idx][0] = clf

        self.fitted = True

    def _joint_log_likelihood(self, X):
        """
        Computes the join log likelihood for the different classifiers
        :param X:
        :return:
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted.")

        _probs = np.zeros((X.shape[0], self.classes.shape[0]))

        for idx in range(len(self.classifier_map)):
            clf, keys = self.classifier_map[idx]
            data = self._get_data(keys, X)
            _probs += clf._joint_log_likelihood(data)

        return _probs

    def predict_log_proba(self, X):
        if not self.fitted:
            raise ValueError("Model has not been fitted.")

        _probs = self._joint_log_likelihood(X)
        # normalize (see naive_bayes.py in sklearn for explanation!!)
        _log_prob_x = logsumexp(_probs, axis=1)
        return _probs - np.atleast_2d(_log_prob_x).T

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model has not been fitted.")
        jll = self._joint_log_likelihood(X)
        return self.classes[np.argmax(jll, axis=1)]

    def serialize(self):
        return {
            "fitted": self.fitted,
            "classes": self.classes,
            "classifier_map": self.classifier_map,
        }

    @staticmethod
    def load(serialized):
        if type(serialized) is not dict:
            raise ValueError("Serialized model has to be a dict")

        fitted = serialized.get('fitted', None)
        classes = serialized.get('classes', None)
        classifier_map = serialized.get('classifier_map', None)

        if fitted is None or classifier_map is None:
            raise ValueError("Missing required serialization entities")

        clf = ThesisSparseNaiveBayes(classifier_map, classes)
        clf.fitted = fitted

        return clf
