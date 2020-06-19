import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate
from sklearn import naive_bayes
import json
import os
import joblib
from timeit import default_timer as timer
from thesislib.utils.ml import models, report
import logging
import pathlib
import numpy as np


def train_rf(data_file, symptoms_db_json, output_dir, rfparams=None, name="", location="QCE", is_nlice=False):
    logger = report.Logger("Random Forest %s Classification on %s" % (name, location))

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if rfparams is None or not isinstance(rfparams, models.RFParams):
        rfparams = models.RFParams()

    try:
        logger.log("Starting Random Forest Classification")
        begin = timer()
        with open(symptoms_db_json) as fp:
            symptoms_db = json.load(fp)
            num_symptoms = len(symptoms_db)

        logger.log("Reading CSV")
        start = timer()
        df = pd.read_csv(data_file, index_col='Index')
        end = timer()
        logger.log("Reading CSV: %.5f secs" % (end - start))

        classes = df.LABEL.unique().tolist()

        logger.log("Prepping Sparse Representation")
        start = timer()
        label_values = df.LABEL.values
        ordered_keys = ['GENDER', 'RACE', 'AGE', 'SYMPTOMS']
        df = df[ordered_keys]

        if is_nlice:
            sparsifier = models.ThesisAIMEDSymptomSparseMaker(num_symptoms=num_symptoms)
        else:
            sparsifier = models.ThesisSymptomSparseMaker(num_symptoms=num_symptoms)
        data_csc = sparsifier.fit_transform(df)

        end = timer()
        logger.log("Prepping Sparse Representation: %.5f secs" % (end - start))

        logger.log("Shuffling Data")
        start = timer()
        split_t = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        train_data = None
        train_labels = None
        test_data = None
        test_labels = None
        for train_index, test_index in split_t.split(data_csc, label_values):
            train_data = data_csc[train_index]
            train_labels = label_values[train_index]
            test_data = data_csc[test_index]
            test_labels = label_values[test_index]

        end = timer()
        logger.log("Shuffling Data: %.5f secs" % (end - start))

        logger.log("Training RF Classifier")
        start = timer()
        clf = RandomForestClassifier(
            n_estimators=rfparams.n_estimators,
            criterion=rfparams.criterion,
            max_depth=rfparams.max_depth,
            min_samples_split=rfparams.min_samples_split,
            min_samples_leaf=rfparams.min_samples_leaf,
            min_weight_fraction_leaf=0.0,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            bootstrap=True,
            oob_score=False,
            n_jobs=2,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None
        )

        clf = clf.fit(train_data, train_labels)
        end = timer()
        logger.log("Training RF Classifier: %.5f secs" % (end - start))

        start = timer()

        scorers = report.get_tracked_metrics(classes=classes, metric_name=[
            report.ACCURACY_SCORE,
            report.PRECISION_WEIGHTED,
            report.RECALL_WEIGHTED,
            report.TOP5_SCORE
        ])

        train_results = {
            "name": "Random Forest",
        }

        for key, scorer in scorers.items():
            logger.log("Starting Score: %s" % key)
            scorer_timer_train = timer()
            train_score = scorer(clf, train_data, train_labels)
            scorer_timer_test = timer()
            test_score = scorer(clf, test_data, test_labels)
            train_results[key] = {
                "train": train_score,
                "test": test_score
            }
            scorer_timer_end = timer()
            train_duration = scorer_timer_test - scorer_timer_train
            test_duration = scorer_timer_end - scorer_timer_test
            duration = scorer_timer_end - scorer_timer_train
            logger.log("Finished score: %s.\nTook: %.5f seconds\nTrain: %.5f, %.5f secs\n Test: %.5f, %.5f secs"
                       % (key, duration, train_score, train_duration, test_score, test_duration))

        end = timer()
        logger.log("Calculating Accuracy: %.5f secs" % (end - start))

        train_results_file = os.path.join(output_dir, "rf_train_results_sparse_grid_search_best.json")
        with open(train_results_file, "w") as fp:
            json.dump(train_results, fp)

        estimator_serialized = {
            "clf": clf,
            "name": "random forest classifier on sparse"
        }
        estimator_serialized_file = os.path.join(output_dir, "rf_serialized_sparse_grid_search_best.joblib")
        joblib.dump(estimator_serialized, estimator_serialized_file)

        finish = timer()
        logger.log("Completed Random Forest Classification: %.5f secs" % (finish - begin))
        res = True
    except Exception as e:
        message = e.__str__()
        logger.log(message, logging.ERROR)
        res = False

    return res


def train_nb(data_file, symptoms_db_json, output_dir, name="", location="QCE", is_nlice=False):
    logger = report.Logger("Naive Bayes %s Classification on %s" %(name, location))

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        message = "Starting Naive Bayes Classification"
        logger.log(message)
        begin = timer()
        with open(symptoms_db_json) as fp:
            symptoms_db = json.load(fp)
            num_symptoms = len(symptoms_db)

        logger.log("Reading CSV")
        start = timer()
        data = pd.read_csv(data_file, index_col='Index')
        end = timer()
        logger.log("Reading CSV: %.5f secs" % (end - start))

        classes = data.LABEL.unique().tolist()

        logger.log("Prepping Sparse Representation")
        start = timer()
        label_values = data.LABEL.values
        ordered_keys = ['GENDER', 'RACE', 'AGE', 'SYMPTOMS']
        data = data[ordered_keys]

        if is_nlice:
            sparsifier = models.ThesisAIMEDSymptomSparseMaker(num_symptoms=num_symptoms)
        else:
            sparsifier = models.ThesisSymptomSparseMaker(num_symptoms=num_symptoms)
        data = sparsifier.fit_transform(data)

        end = timer()
        logger.log("Prepping Sparse Representation: %.5f secs" % (end - start))

        logger.log("Shuffling Data")
        start = timer()
        split_t = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        train_data = None
        train_labels = None
        test_data = None
        test_labels = None
        for train_index, test_index in split_t.split(data, label_values):
            train_data = data[train_index]
            train_labels = label_values[train_index]
            test_data = data[test_index]
            test_labels = label_values[test_index]

        end = timer()
        logger.log("Shuffling Data: %.5f secs" % (end - start))

        logger.log("Training Naive Bayes")
        start = timer()
        gender_clf = naive_bayes.BernoulliNB()
        race_clf = naive_bayes.MultinomialNB()
        age_clf = naive_bayes.GaussianNB()
        symptom_clf = naive_bayes.BernoulliNB()

        if not is_nlice:
            classifier_map = [
                [gender_clf, [0, False]],
                [race_clf, [1, False]],
                [age_clf, [2, False]],
                [symptom_clf, [(3, None), True]],
            ]
        else:
            symptom_nlice_clf = naive_bayes.GaussianNB()
            reg_indices = [0, 1, 2] + [9, 12, 20, 25]
            bern_indices = []
            for idx in range(train_data.shape[1]):
                if idx not in reg_indices:
                    bern_indices.append(idx)
            new_indices = reg_indices + bern_indices
            train_data = train_data[:, new_indices]
            test_data = test_data[:, new_indices]
            classifier_map = [
                [gender_clf, [0, False]],
                [race_clf, [1, False]],
                [age_clf, [2, False]],
                [symptom_nlice_clf, [(3, 7), False]],
                [symptom_clf, [(7, None), True]]
            ]

        clf = models.ThesisSparseNaiveBayes(classifier_map=classifier_map, classes=classes)

        clf.fit(train_data, train_labels)
        end = timer()
        logger.log("Training Naive Classifier: %.5f secs" % (end - start))

        logger.log("Calculating Accuracy")
        start = timer()

        scorers = report.get_tracked_metrics(classes=classes)
        train_results = {
            "name": "Naive Bayes Classifier",
        }

        for key, scorer in scorers.items():
            logger.log("Starting Score: %s" % key)
            scorer_timer_train = timer()
            train_score = scorer(clf, train_data, train_labels)
            scorer_timer_test = timer()
            test_score = scorer(clf, test_data, test_labels)
            train_results[key] = {
                "train": train_score,
                "test": test_score
            }
            scorer_timer_end = timer()
            train_duration = scorer_timer_test - scorer_timer_train
            test_duration = scorer_timer_end - scorer_timer_test
            duration = scorer_timer_end - scorer_timer_train
            logger.log("Finished score: %s.\nTook: %.5f seconds\nTrain: %.5f, %.5f secs\n Test: %.5f, %.5f secs"
                       % (key, duration, train_score, train_duration, test_score, test_duration))

        end = timer()
        logger.log("Calculating Accuracy: %.5f secs" % (end - start))

        train_results_file = os.path.join(output_dir, "nb_train_results_sparse.json")
        with open(train_results_file, "w") as fp:
            json.dump(train_results, fp, indent=4)

        estimator_serialized = {
            "clf": clf.serialize(),
            "name": "naive bayes classifier on sparse"
        }
        estimator_serialized_file = os.path.join(output_dir, "nb_serialized_sparse.joblib")
        joblib.dump(estimator_serialized, estimator_serialized_file)

        finish = timer()
        logger.log("Completed Naive Classification: %.5f secs" % (finish - begin))
        res = True
    except Exception as e:
        raise e
        message = e.__str__()
        logger.log(message, logging.ERROR)
        res = False

    return res


def train_ai_med_rf(data_file, symptoms_db_json, output_dir, rfparams=None, name="", location="QCE", is_nlice=False):
    logger = report.Logger("Random Forest %s Classification on %s" % (name, location))

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if rfparams is None or not isinstance(rfparams, models.RFParams):
        rfparams = models.RFParams()

    try:
        logger.log("Starting Random Forest Classification")
        begin = timer()
        with open(symptoms_db_json) as fp:
            symptoms_db = json.load(fp)
            num_symptoms = len(symptoms_db)

        logger.log("Reading CSV")
        start = timer()
        df = pd.read_csv(data_file, index_col='Index')
        end = timer()
        logger.log("Reading CSV: %.5f secs" % (end - start))

        classes = df.LABEL.unique().tolist()

        logger.log("Prepping Sparse Representation")
        start = timer()
        label_values = df.LABEL.values
        ordered_keys = ['GENDER', 'RACE', 'AGE', 'SYMPTOMS']
        df = df[ordered_keys]

        if is_nlice:
            sparsifier = models.ThesisAIMEDSymptomSparseMaker(num_symptoms=num_symptoms)
        else:
            sparsifier = models.ThesisSymptomSparseMaker(num_symptoms=num_symptoms)

        data_csc = sparsifier.fit_transform(df)
        end = timer()
        logger.log("Prepping Sparse Representation: %.5f secs" % (end - start))

        logger.log("Running RF Classifier Cross Validate")
        start = timer()
        clf = RandomForestClassifier(
            n_estimators=rfparams.n_estimators,
            criterion=rfparams.criterion,
            max_depth=rfparams.max_depth,
            min_samples_split=rfparams.min_samples_split,
            min_samples_leaf=rfparams.min_samples_leaf,
            min_weight_fraction_leaf=0.0,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            bootstrap=True,
            oob_score=False,
            n_jobs=2,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None
        )

        scorers = report.get_tracked_metrics(classes=classes, metric_name=[
            report.ACCURACY_SCORE,
            report.PRECISION_WEIGHTED,
            report.RECALL_WEIGHTED,
            report.TOP5_SCORE
        ])

        cv_res = cross_validate(
            clf,
            data_csc,
            label_values,
            scoring=scorers,
            return_train_score=True,
            return_estimator=True,
            error_score='raise'
        )

        end = timer()
        logger.log("Running RF Classifier Cross Validate: %.5f secs" % (end - start))

        train_results = {
            "name": "Random Forest",
        }

        abs_test_score = None
        for key in scorers.keys():
            train_score = np.mean(cv_res["train_%s" % key])
            test_score = np.mean(cv_res["test_%s" % key])
            train_results[key] = {
                "train": train_score,
                "test": test_score
            }
            logger.log("RF Finished score: %s.\nTrain: %.5f\nTest: %.5f"
                       % (key, train_score, test_score))
            if key == report.ACCURACY_SCORE:
                abs_test_score = np.abs(cv_res["train_%s" % key] - test_score)

        end = timer()
        logger.log("Calculating Accuracy: %.5f secs" % (end - start))

        train_results_file = os.path.join(output_dir, "rf_train_results_sparse_grid_search_best.json")
        with open(train_results_file, "w") as fp:
            json.dump(train_results, fp)

        finish = timer()
        logger.log("Completed Random Forest Classification: %.5f secs" % (finish - begin))

        # save model
        if abs_test_score is not None:
            estimator_idx = np.argmin(abs_test_score)  # pick the closest to the average
        else:
            estimator_idx = 0  # pick the first one

        clf = cv_res['estimator'][estimator_idx]
        estimator_serialized = {
            "clf": clf,
            "name": "random forest classifier on sparse"
        }

        nlice_suffix = "_nlice" if is_nlice else ""
        estimator_serialized_file = os.path.join(output_dir, "rf_serialized_sparse%s.joblib" % nlice_suffix)
        joblib.dump(estimator_serialized, estimator_serialized_file)

        res = True
    except Exception as e:
        message = e.__str__()
        logger.log(message, logging.ERROR)
        res = False

    return res


def train_ai_med_nb(
        data_file,
        symptoms_db_json,
        output_dir, name="",
        location="QCE",
        is_nlice=False,
        nlice_symptoms=None,
        nlice_symptoms_enc=None
):
    logger = report.Logger("Naive Bayes %s Classification on %s" %(name, location))

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        message = "Starting Naive Bayes Classification"
        logger.log(message)
        begin = timer()
        with open(symptoms_db_json) as fp:
            symptoms_db = json.load(fp)
            num_symptoms = len(symptoms_db)

        if is_nlice:
            symptom_vector = sorted(symptoms_db.keys())
            sorted_symptoms = {key: idx for idx, key in enumerate(symptom_vector)}
            nlice_indices = [sorted_symptoms[key] + 3 for key in nlice_symptoms]
            nlice_encoding = [nlice_symptoms_enc[key] for key in nlice_symptoms]
        else:
            nlice_indices = []
            nlice_encoding = []

        logger.log("Reading CSV")
        start = timer()
        data = pd.read_csv(data_file, index_col='Index')
        end = timer()
        logger.log("Reading CSV: %.5f secs" % (end - start))

        classes = data.LABEL.unique().tolist()

        logger.log("Prepping Sparse Representation")
        start = timer()
        label_values = data.LABEL.values
        ordered_keys = ['GENDER', 'RACE', 'AGE', 'SYMPTOMS']
        data = data[ordered_keys]

        if is_nlice:
            sparsifier = models.ThesisAIMEDSymptomSparseMaker(num_symptoms=num_symptoms)
        else:
            sparsifier = models.ThesisSymptomSparseMaker(num_symptoms=num_symptoms)
        data = sparsifier.fit_transform(data)

        end = timer()
        logger.log("Prepping Sparse Representation: %.5f secs" % (end - start))

        logger.log("Cross validate on Training Naive Bayes")
        start = timer()
        gender_clf = naive_bayes.BernoulliNB()
        race_clf = models.ThesisCategoricalNB()
        age_clf = naive_bayes.GaussianNB()
        symptom_clf = naive_bayes.BernoulliNB()

        if not is_nlice:
            classifier_map = [
                [gender_clf, [0, False]],
                [race_clf, [1, False]],
                [age_clf, [2, False]],
                [symptom_clf, [(3, None), True]],
            ]
        else:
            symptom_nlice_clf = models.ThesisCategoricalNB(skip_zero=True, encoding_map=nlice_encoding)
            plain_indices = [0, 1, 2]
            reg_indices = plain_indices + nlice_indices
            nlice_index_end = 3 + len(nlice_indices)
            bern_indices = []
            for idx in range(data.shape[1]):
                if idx not in reg_indices:
                    bern_indices.append(idx)
            new_indices = reg_indices + bern_indices
            data = data[:, new_indices]
            classifier_map = [
                [gender_clf, [0, False]],
                [race_clf, [1, False]],
                [age_clf, [2, False]],
                [symptom_nlice_clf, [(3, nlice_index_end), False]],
                [symptom_clf, [(nlice_index_end, None), True]]
            ]

        clf = models.ThesisSparseNaiveBayes(classifier_map=classifier_map, classes=classes)

        scorers = report.get_tracked_metrics(classes=classes, metric_name=[
            report.ACCURACY_SCORE,
            report.PRECISION_WEIGHTED,
            report.RECALL_WEIGHTED,
            report.TOP5_SCORE
        ])

        cv_res = cross_validate(
            clf,
            data,
            label_values,
            scoring=scorers,
            return_train_score=True,
            return_estimator=True,
            error_score='raise'
        )

        train_results = {
            "name": "Naive Bayes Classifier AI MED",
        }

        abs_test_score = None

        for key in scorers.keys():
            train_score = np.mean(cv_res["train_%s" % key])
            test_score = np.mean(cv_res["test_%s" % key])
            train_results[key] = {
                "train": train_score,
                "test": test_score
            }
            if key == report.ACCURACY_SCORE:
                abs_test_score = np.abs(cv_res["test_%s" % key] - test_score)

            logger.log("NB Finished score: %s.\nTrain: %.5f\nTest: %.5f"
                       % (key, train_score, test_score))

        end = timer()
        logger.log("Calculating Accuracy: %.5f secs" % (end - start))

        train_results_file = os.path.join(output_dir, "nb_train_results_sparse.json")
        with open(train_results_file, "w") as fp:
            json.dump(train_results, fp, indent=4)

        finish = timer()
        logger.log("Completed Naive Classification: %.5f secs" % (finish - begin))

        # save model
        if abs_test_score is not None:
            estimator_idx = np.argmin(abs_test_score) # pick the closest to the average
        else:
            estimator_idx = 0 # pick the first one

        clf = cv_res['estimator'][estimator_idx]
        estimator_serialized = {
            "clf": clf,
            "name": "naive bayes classifier on sparse"
        }

        nlice_suffix = "_nlice" if is_nlice else ""
        estimator_serialized_file = os.path.join(output_dir, "nb_serialized_sparse%s.joblib" % nlice_suffix)
        joblib.dump(estimator_serialized, estimator_serialized_file)

        res = True
    except Exception as e:
        raise e
        message = e.__str__()
        logger.log(message, logging.ERROR)
        res = False

    return res
