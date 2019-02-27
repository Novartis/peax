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
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import forestci as fci

from server.utils import unpredictability, uncertainty, convergence, divergence


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


def evaluate_threading(fn, X, X_train, prev_classifier, prev_prev_classifier, done):
    fn(X, X_train, prev_classifier, prev_prev_classifier)
    done()


class Classifier:
    def __init__(self, search_id: int, classifier_id: int, **kwargs):
        self.search_id = search_id
        self.classifier_id = classifier_id
        self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        self.unpredictability = (
            kwargs["unpredictability"] if "unpredictability" in kwargs else None
        )
        self.uncertainty = kwargs["uncertainty"] if "uncertainty" in kwargs else None
        self.convergence = kwargs["convergence"] if "convergence" in kwargs else None
        self.divergence = kwargs["divergence"] if "divergence" in kwargs else None
        self.is_trained = False
        self.is_training = False
        self.is_evaluated = (
            self.unpredictability is not None and self.uncertainty is not None
        )
        self.is_evaluating = False
        self.serialized_classifications = b""

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

    def evaluate(self, X, X_train, prev_classifier=None, prev_prev_classifier=None):
        fit_y, p_y = self.predict(X)

        self.unpredictability = unpredictability(p_y[:, 1])
        self.uncertainty = uncertainty(self.model, X_train, X)

        if prev_classifier is not None and prev_prev_classifier is not None:
            prev_p_y = prev_classifier.model.predict_proba(X)
            prev_prev_p_y = prev_classifier.model.predict_proba(X)

            self.convergence = convergence(
                prev_prev_p_y[:, 1], prev_p_y[:, 1], p_y[:, 1]
            )
            self.divergence = divergence(prev_prev_p_y[:, 1], prev_p_y[:, 1], p_y[:, 1])

        return (
            self.unpredictability,
            self.uncertainty,
            self.convergence,
            self.divergence,
        )

    def evaluate_threading(
        self,
        X,
        X_train,
        prev_classifier=None,
        prev_prev_classifier=None,
        callback: callable = None,
    ):
        self.is_evaluated = False
        self.is_evaluating = True
        print("YASSAS!", self.is_evaluating)
        try:
            _thread.start_new_thread(
                evaluate_threading,
                (
                    self.evaluate,
                    X,
                    X_train,
                    prev_classifier,
                    prev_prev_classifier,
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
