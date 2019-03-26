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
import numpy as np
from server import utils
from server.classifier import Classifier
from server.defaults import MIN_CLASSIFICATIONS
from server.exceptions import LabelsDidNotChange, TooFewLabels


def get_labels(classifier, search_target_windows):
    labeled_windows = np.abs(
        utils.unserialize_classif(classifier.serialized_classifications)
    )

    return np.concatenate((labeled_windows, search_target_windows)).astype(np.int)


class ClassifierNotFound(Exception):
    """Raised when no classifier was found"""

    pass


class Classifiers:
    def __init__(
        self,
        db,
        data,
        window_size: int,
        abs_offset: int,
        min_classifications: int = MIN_CLASSIFICATIONS,
    ):
        self.classifiers = {}
        self.db = db
        self.data = data
        self.window_size = window_size
        self.abs_offset = abs_offset
        self.min_classifications = min_classifications

    def delete(self, search_id: int, classifier_id: int = None):
        self.db.delete_classifier(search_id, classifier_id)
        self.classifiers.pop(search_id, None)

    def get(self, search_id: int, classifier_id: int = None, **kwargs):
        classifier_info = self.db.get_classifier(search_id, classifier_id)

        if classifier_info is None:
            if "default" in kwargs:
                return kwargs["default"]

            raise ClassifierNotFound(
                "No classifier for search #{}{} found".format(
                    search_id,
                    " with id {}".format(classifier_id)
                    if classifier_id is not None
                    else "",
                )
            )

        classifier_id = classifier_info["classifier_id"]

        if (
            search_id in self.classifiers
            and classifier_id in self.classifiers[search_id]
        ):
            return self.classifiers[search_id][classifier_id]

        classifier = Classifier(**classifier_info)

        if classifier_info["model"] is not None:
            classifier.load(classifier_info["model"])

        classifier.serialized_classifications = classifier_info[
            "serialized_classifications"
        ]

        if search_id not in self.classifiers:
            self.classifiers[search_id] = {classifier.classifier_id: classifier}
        else:
            self.classifiers[search_id][classifier.classifier_id] = classifier

        return classifier

    def evaluate(
        self,
        search_id: int,
        classifier_id: int = None,
        update: bool = False,
        no_threading: bool = False,
    ):
        classifier_info = self.db.get_classifier(search_id, classifier_id)

        if classifier_info is None:
            return None

        classifier_id = classifier_info["classifier_id"]
        classifier = self.get(search_id, classifier_id)

        if classifier.is_evaluated and not update:
            return None

        if classifier_info["model"] is not None:
            classifier.load(classifier_info["model"])

        # Get search target classifications
        search_target_windows = utils.get_search_target_windows(
            self.db, search_id, self.window_size, self.abs_offset, no_stack=True
        )

        # Get labels used to train the classifier
        labeled_windows = get_labels(classifier, search_target_windows)

        train = self.data[labeled_windows]
        test = self.data

        prev_classifier = None
        prev_classifier_info = None
        prev_train = None
        prev_prev_classifier = None
        prev_prev_classifier_info = None
        prev_prev_train = None

        if classifier_id >= 1:
            prev_classifier_info = self.db.get_classifier(search_id, classifier_id - 1)

            if prev_classifier_info["model"] is not None:
                prev_classifier = self.get(search_id, classifier_id - 1)
                prev_classifier.load(prev_classifier_info["model"])
                prev_train = self.data[
                    get_labels(prev_classifier, search_target_windows)
                ]

        if classifier_id >= 2:
            prev_prev_classifier_info = self.db.get_classifier(
                search_id, classifier_id - 2
            )

            if prev_prev_classifier_info["model"] is not None:
                prev_prev_classifier = self.get(search_id, classifier_id - 2)
                prev_prev_classifier.load(prev_prev_classifier_info["model"])
                prev_prev_train = self.data[
                    get_labels(prev_prev_classifier, search_target_windows)
                ]

        def set_evaluate_results():
            self.db.set_classifier(
                search_id,
                classifier_id,
                unpredictability_all=classifier.unpredictability_all,
            )
            self.db.set_classifier(
                search_id,
                classifier_id,
                unpredictability_labels=classifier.unpredictability_labels,
            )
            self.db.set_classifier(
                search_id,
                classifier_id,
                prediction_proba_change_all=classifier.prediction_proba_change_all,
            )
            self.db.set_classifier(
                search_id,
                classifier_id,
                prediction_proba_change_labels=classifier.prediction_proba_change_labels,
            )
            self.db.set_classifier(
                search_id, classifier_id, convergence_all=classifier.convergence_all
            )
            self.db.set_classifier(
                search_id,
                classifier_id,
                convergence_labels=classifier.convergence_labels,
            )
            self.db.set_classifier(
                search_id, classifier_id, divergence_all=classifier.divergence_all
            )
            self.db.set_classifier(
                search_id, classifier_id, divergence_labels=classifier.divergence_labels
            )

        if no_threading:
            classifier.evaluate(
                test,
                train,
                prev_classifier=prev_classifier,
                prev_train=prev_train,
                prev_prev_classifier=prev_prev_classifier,
                prev_prev_train=prev_prev_train,
            )
            set_evaluate_results()
        else:
            classifier.evaluate_threading(
                test,
                train,
                prev_classifier=prev_classifier,
                prev_train=prev_train,
                prev_prev_classifier=prev_prev_classifier,
                prev_prev_train=prev_prev_train,
                callback=set_evaluate_results,
            )

    def evaluate_all(self, search_id: int, update: bool = False):
        classifier_ids = self.db.get_classifier_ids(search_id)

        if classifier_ids is None:
            return None

        print("wurst puller")
        for classifier_id in classifier_ids:
            self.evaluate(search_id, classifier_id, update=update, no_threading=True)

    def evaluate_all_threading(self, search_id: int, update: bool = False):
        try:
            _thread.start_new_thread(self.evaluate_all, (search_id, update))
        except Exception:
            print("Evaluation failed")

    def new(self, search_id: int):
        # Get previous classifier
        prev_classifier = self.get(search_id)
        prev_classif = None
        if prev_classifier is not None:
            prev_classif = prev_classifier.serialized_classifications

        # Get search target classifications
        search_target_classif = utils.get_search_target_classif(
            self.db, search_id, self.window_size, self.abs_offset
        )

        dbres = self.db.get_classifications(search_id)

        N = self.data.shape[0]

        classifications = np.array(
            list(map(lambda x: [int(x["windowId"]), int(x["classification"])], dbres))
        )

        # Serialize classifications
        new_classif = utils.serialize_classif(classifications)

        # Compare new classifications with old classifications
        if new_classif == prev_classif:
            raise LabelsDidNotChange()

        # Do not train a classifier if less than the minimum of labels are available
        if classifications.size < self.min_classifications:
            raise TooFewLabels(
                {
                    "num_labels": classifications.size,
                    "min_classifications": self.min_classifications,
                }
            )

        # Create a DB entry
        classifier_id = self.db.create_classifier(search_id, classif=new_classif)

        # Combine classifications with search target
        if np.min(search_target_classif) >= 0 and np.max(search_target_classif) < N:
            classifications = np.vstack((search_target_classif, classifications))

        # Change `-1` to `0`
        classifications[:, 1][np.where(classifications[:, 1] == -1)] = 0

        train_X = self.data[classifications[:, 0]]
        train_y = classifications[:, 1]

        classifier = self.get(search_id, classifier_id)
        classifier.serialized_classifications = new_classif

        if search_id not in self.classifiers:
            self.classifiers[search_id] = {classifier.classifier_id: classifier}
        else:
            self.classifiers[search_id][classifier.classifier_id] = classifier

        def callback_train():
            # Dump and store the trained model
            dumped_model = classifier.dump()
            self.db.set_classifier(search_id, classifier_id, model=dumped_model)

            # Evaluate classifier
            self.evaluate(search_id, classifier_id)

        classifier.train(train_X, train_y, callback=callback_train)

        return classifier
