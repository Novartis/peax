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

import numpy as np
from server import utils
from server.classifier import Classifier


class Classifiers:

    def __init__(
        self, db, data, chromsizes_path: str, window_size: int, abs_offset: int
    ):
        self.classifiers = {}
        self.db = db
        self.data = data
        self.chromsizes_path = chromsizes_path
        self.window_size = window_size
        self.abs_offset = abs_offset

    def delete(self, search_id: int, classifier_id: int = None):
        self.db.delete_classifier(search_id, classifier_id)
        self.classifiers.pop(search_id, None)

    def get(self, search_id: int, classifier_id: int = None):
        if search_id in self.classifiers:
            return self.classifiers[search_id]

        classifier_info = self.db.get_classifier(search_id, classifier_id)

        if classifier_info is not None:
            classifier = Classifier(
                search_id, classifier_info["classifier_id"]
            )
            if classifier_info["model"] is not None:
                classifier.load(classifier_info["model"])
            classifier.serialized_classifications = classifier_info[
                "serialized_classifications"
            ]
            self.classifiers[search_id] = classifier
            return classifier

        return None

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
            list(
                map(
                    lambda x: [int(x["windowId"]), int(x["classification"])],
                    dbres,
                )
            )
        )

        # Serialize classifications
        new_classif = utils.serialize_classif(classifications)

        # Compare new classifications with old classifications
        if new_classif == prev_classif:
            return None

        # Create a DB entry
        classifier_id = self.db.create_classifier(
            search_id, classif=new_classif
        )

        # Combine classifications with search target
        if (
            np.min(search_target_classif) >= 0
            and np.max(search_target_classif) < N
        ):
            classifications = np.vstack(
                (search_target_classif, classifications)
            )

        # Change `-1` to `0`
        classifications[:, 1][np.where(classifications[:, 1] == -1)] = 0

        train_X = self.data[classifications[:, 0]]
        train_y = classifications[:, 1]

        classifier = Classifier(search_id, classifier_id)
        classifier.serialized_classifications = new_classif
        self.classifiers[search_id] = classifier

        def callback():
            # Store the trained model
            dumped_model = classifier.dump()
            self.db.set_classifier(search_id, classifier_id, dumped_model)

        classifier.train(train_X, train_y, callback=callback)

        return classifier
