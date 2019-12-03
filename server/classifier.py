"""
Copyright 2018 Novartis Institutes for BioMedical Research Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import _thread
import joblib

from io import BytesIO

from sklearn.utils.testing import all_estimators

estimators = all_estimators()

from server.utils import (
    unpredictability,
    prediction_proba_change,
    convergence,
    divergence,
)


def test_classifier(Classifier):
    return hasattr(Classifier, "fit") and hasattr(Classifier, "predict_proba")


available_sklearn_classifiers = {}
for name, Classifier in estimators:
    if test_classifier(Classifier):
        available_sklearn_classifiers[name] = Classifier


def get_classifier(classifier_name):
    if classifier_name in available_sklearn_classifiers:
        return available_sklearn_classifiers[classifier_name]
    return None


def done(instance, prefix: str, callback: callable = None):
    def wrapped():
        setattr(instance, "{}ed".format(prefix), True)
        setattr(instance, "{}ing".format(prefix), False)
        if callback is not None:
            callback()

    return wrapped


def train_threading(fit, train_X, train_y, done):
    fit(train_X, train_y)
    done()


def evaluate_threading(
    fn,
    X,
    X_train,
    prev_classifier,
    prev_train,
    prev_prev_classifier,
    prev_prev_train,
    done,
):
    fn(X, X_train, prev_classifier, prev_train, prev_prev_train, prev_prev_classifier)
    done()


class Classifier:
    def __init__(
        self,
        classifier_class: str,
        classifier_params: dict,
        search_id: int,
        classifier_id: int,
        **kwargs,
    ):
        self.search_id = search_id
        self.classifier_id = classifier_id

        if isinstance(classifier_class, str):
            if get_classifier(classifier_class):
                self.model = get_classifier(classifier_class)(**classifier_params)
            else:
                raise ValueError(
                    f"Unknown or unsupported classifier: {classifier_class}"
                )
        else:
            if test_classifier(classifier_class):
                self.model = classifier_class(**classifier_params)
            else:
                raise ValueError(
                    "Custom classifier needs to support fit and predict_proba"
                )

        try:
            self.unpredictability_all = kwargs["unpredictability_all"]
        except KeyError:
            self.unpredictability_all = None
        try:
            self.unpredictability_labels = kwargs["unpredictability_labels"]
        except KeyError:
            self.unpredictability_labels = None
        try:
            self.prediction_proba_change_all = kwargs["prediction_proba_change_all"]
        except KeyError:
            self.prediction_proba_change_all = None
        try:
            self.prediction_proba_change_labels = kwargs[
                "prediction_proba_change_labels"
            ]
        except KeyError:
            self.prediction_proba_change_labels = None
        try:
            self.convergence_all = kwargs["convergence_all"]
        except KeyError:
            self.convergence_all = None
        try:
            self.convergence_labels = kwargs["convergence_labels"]
        except KeyError:
            self.convergence_labels = None
        try:
            self.divergence_all = kwargs["divergence_all"]
        except KeyError:
            self.divergence_all = None
        try:
            self.divergence_labels = kwargs["divergence_labels"]
        except KeyError:
            self.divergence_labels = None

        self.is_trained = False
        self.is_training = False
        self.is_evaluated = (
            self.unpredictability_all is not None
            and self.unpredictability_labels is not None
        )
        self.is_evaluating = False
        self.serialized_classifications = (
            kwargs["serialized_classifications"]
            if "serialized_classifications" in kwargs
            else b""
        )

    def predict(self, X):
        if not self.is_trained:
            return None, None

        fit_y = self.model.predict(X)
        p_y = self.model.predict_proba(X)

        return fit_y, p_y

    def train(
        self, train_X, train_y, n_estimators: int = 100, callback: callable = None
    ):
        self.is_trained = False
        self.is_training = True
        try:
            _thread.start_new_thread(
                train_threading,
                (self.model.fit, train_X, train_y, done(self, "is_train", callback)),
            )
        except Exception:
            self.is_trained = False
            self.is_training = False

    def evaluate(
        self,
        X,
        train,
        prev_classifier=None,
        prev_train=None,
        prev_prev_classifier=None,
        prev_prev_train=None,
    ):
        p_y_all = self.model.predict_proba(X)[:, 1]
        p_y_labels = self.model.predict_proba(train)[:, 1]

        self.unpredictability_all = unpredictability(p_y_all)
        self.unpredictability_labels = unpredictability(p_y_labels)

        if prev_classifier is not None:
            prev_p_y_all = prev_classifier.model.predict_proba(X)[:, 1]
            p_y_prev_labels = self.model.predict_proba(prev_train)[:, 1]
            prev_p_y_labels = prev_classifier.model.predict_proba(prev_train)[:, 1]

            self.prediction_proba_change_all = prediction_proba_change(
                p_y_all, prev_p_y_all
            )
            self.prediction_proba_change_labels = prediction_proba_change(
                p_y_prev_labels, prev_p_y_labels
            )

            if prev_prev_classifier is not None:
                prev_prev_p_y_all = prev_prev_classifier.model.predict_proba(X)[:, 1]
                p_y_prev_prev_labels = self.model.predict_proba(prev_prev_train)[:, 1]
                prev_p_y_prev_labels = prev_classifier.model.predict_proba(
                    prev_prev_train
                )[:, 1]
                prev_prev_p_y_labels = prev_prev_classifier.model.predict_proba(
                    prev_prev_train
                )[:, 1]

                self.convergence_all = convergence(
                    prev_prev_p_y_all, prev_p_y_all, p_y_all
                )
                self.convergence_labels = convergence(
                    prev_prev_p_y_labels, prev_p_y_prev_labels, p_y_prev_prev_labels
                )

                self.divergence_all = divergence(
                    prev_prev_p_y_all, prev_p_y_all, p_y_all
                )
                self.divergence_labels = divergence(
                    prev_prev_p_y_labels, prev_p_y_prev_labels, p_y_prev_prev_labels
                )

        return (
            self.unpredictability_all,
            self.unpredictability_labels,
            self.prediction_proba_change_all,
            self.prediction_proba_change_labels,
            self.convergence_all,
            self.convergence_labels,
            self.divergence_all,
            self.divergence_labels,
        )

    def evaluate_threading(
        self,
        X,
        train,
        prev_classifier=None,
        prev_train=None,
        prev_prev_classifier=None,
        prev_prev_train=None,
        callback: callable = None,
    ):
        self.is_evaluated = False
        self.is_evaluating = True
        try:
            # fn, X, X_train, prev_classifier, prev_prev_classifier, done

            _thread.start_new_thread(
                evaluate_threading,
                (
                    self.evaluate,
                    X,
                    train,
                    prev_classifier,
                    prev_train,
                    prev_prev_classifier,
                    prev_prev_train,
                    done(self, "is_evaluat", callback),
                ),
            )
        except Exception:
            self.is_evaluated = False
            self.is_evaluating = False

    def load(self, dumped_model):
        with BytesIO(dumped_model) as b:
            self.model = joblib.load(b)
            self.is_trained = True

    def dump(self):
        with BytesIO() as b:
            joblib.dump(self.model, b)
            return b.getvalue()
