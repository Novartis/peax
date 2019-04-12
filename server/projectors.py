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

import json
import numpy as np
from server import utils
from server import projector

Projector = projector.Projector
DEFAULT_N_NEIGHBORS = projector.DEFAULT_PROJECTOR_SETTINGS["n_neighbors"]
DEFAULT_MIN_DIST = projector.DEFAULT_PROJECTOR_SETTINGS["min_dist"]


class Projectors:
    def __init__(self, db, data, window_size, abs_offset):
        self.projectors = {}
        self.db = db
        self.data = data
        self.window_size = window_size
        self.abs_offset = abs_offset

    def delete(self, search_id: int, projector_id: int = None):
        self.db.delete_projector(search_id, projector_id)
        self.projectors.pop(search_id, None)

    def fit(
        self,
        search_id: int,
        projector_id: int,
        projector=None,
        classifications: np.ndarray = None,
    ):
        if projector is None:
            projector = self.get(search_id, projector_id)

        if projector.is_fitting or projector.is_fitted:
            return

        if classifications is None:
            classifications = self.getClassifications(search_id)

        X, y = self.getXY(search_id, classifications)

        def projected():
            # Store the projection
            self.db.set_projector(
                search_id,
                projector.projector_id,
                projection=projector.projection.tobytes(),
            )

        def fitted():
            # Store the projector model
            self.db.set_projector(
                search_id, projector.projector_id, projector=projector.dump()
            )
            projector.project(X, callback=projected)

        projector.fit(X, y, callback=fitted)

    def get(self, search_id: int, projector_id: int = None):
        if search_id in self.projectors:
            return self.projectors[search_id]

        proj_info = self.db.get_projector(search_id, projector_id)

        if proj_info is not None:
            projector = Projector(search_id, proj_info["projector_id"])

            if proj_info["projector"]:
                projector.load(proj_info["projector"])
                projector.is_fitted = True

            if proj_info["projection"]:
                projector.projection = np.frombuffer(
                    proj_info["projection"], np.float32
                ).reshape(-1, 2)

            if proj_info["classifications"]:
                projector.classifications = proj_info["classifications"]

            if proj_info["settings"]:
                projector.settings = json.loads(proj_info["settings"])

            self.projectors[search_id] = projector
            return projector

        return None

    def getClassifications(self, search_id):
        return np.array(
            list(
                map(
                    lambda x: [int(x["windowId"]), int(x["classification"])],
                    self.db.get_classifications(search_id),
                )
            )
        )

    def getXY(self, search_id: int, classifications: np.ndarray):
        N = self.data.shape[0]

        # Get search target classifications
        search_target_classif = utils.get_search_target_classif(
            self.db, search_id, self.window_size, self.abs_offset
        )

        # Combine classifications with search target
        if np.min(search_target_classif) >= 0 and np.max(search_target_classif) < N:
            if classifications.size == 0:
                classifications = search_target_classif
            else:
                classifications = np.vstack((search_target_classif, classifications))

        unclassified = np.where(classifications[:, 1] == 0)
        uninteresting = np.where(classifications[:, 1] == -1)

        # Change `0` to `-1` as `-1` is the standard encoding in sklearn for
        # unlabeled
        classifications[unclassified, 1] = -1

        # Change `-1` to `0`
        classifications[uninteresting, 1] = 0

        total_classification = np.zeros(N)
        total_classification[:] = -1
        total_classification[classifications[:, 0]] = classifications[:, 1]

        return (self.data, total_classification)

    def new(
        self,
        search_id: int,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
        min_dist: float = DEFAULT_MIN_DIST,
    ):
        # Get previous projector
        prev_projector = self.get(search_id)
        prev_classif = None
        prev_settings = None
        if prev_projector is not None:
            prev_classif = prev_projector.classifications
            prev_settings = prev_projector.settings

        classifications = self.getClassifications(search_id)

        # Serialize classifications
        new_classif = (
            b""
            if classifications.size == 0
            else utils.serialize_classif(classifications)
        )

        # Compare new classifications with old classifications
        if (
            new_classif == prev_classif
            and n_neighbors == prev_settings["n_neighbors"]
            and min_dist == prev_settings["min_dist"]
        ):
            return prev_projector

        # Create a DB entry
        projector_id = self.db.create_projector(search_id, classifications=new_classif)

        projector = Projector(
            search_id, projector_id, n_neighbors=n_neighbors, min_dist=min_dist
        )
        self.db.set_projector(
            search_id, projector_id, settings=json.dumps(projector.settings)
        )
        projector.classifications = new_classif
        self.projectors[search_id] = projector

        # For the projector
        self.fit(
            search_id,
            projector_id,
            projector=projector,
            classifications=classifications,
        )

        return projector
