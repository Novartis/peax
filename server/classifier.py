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


def done(instance, callback=None):
    def wrapped():
        instance.is_trained = True
        instance.is_training = False
        if callback is not None:
            callback()

    return wrapped


def train_threaded(fit, train_X, train_y, done):
    fit(train_X, train_y)
    done()


class Classifier:
    def __init__(self, search_id: int, classifier_id: int):
        self.search_id = search_id
        self.classifier_id = classifier_id
        self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        self.is_trained = False
        self.is_training = False
        self.serialized_classifications = b""

    def predict(self, X):
        if not self.is_trained:
            return None, None

        fit_y = self.model.predict(X)
        p_y = self.model.predict_proba(X)

        return fit_y, p_y

    def train(self, train_X, train_y, n_estimators: int = 100, callback=None):
        self.is_trained = False
        self.is_training = True
        try:
            _thread.start_new_thread(
                train_threaded, (self.model.fit, train_X, train_y, done(self, callback))
            )
        except Exception:
            self.is_trained = False
            self.is_training = False

    def load(self, dumped_model):
        with BytesIO(dumped_model) as b:
            self.model = joblib.load(b)
            self.is_trained = True

    def dump(self):
        with BytesIO() as b:
            joblib.dump(self.model, b)
            return b.getvalue()
