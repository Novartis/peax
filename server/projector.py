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
import umap
from io import BytesIO
from sklearn.externals import joblib

DEFAULT_PROJECTOR = umap.UMAP
DEFAULT_PROJECTOR_SETTINGS = {"n_neighbors": 5, "min_dist": 0.1, "metric": "euclidean"}


def normalize(projection):
    min_vals = np.min(projection, axis=0)
    # Move to origin (0, 0)
    projection -= min_vals
    max_vals = np.max(projection, axis=0)
    # Normalize to [-1, 1]
    projection = projection / (max_vals / 2) - 1

    return projection.astype(np.float32)


def fitted(ctx, callback=None):
    def wrapped(*args):
        ctx.is_fitted = True
        ctx.is_fitting = False
        if callback is not None:
            callback()

    return wrapped


def projected(ctx, callback=None):
    def wrapped(projection):
        ctx.projection = normalize(projection)
        if callback is not None:
            callback()

    return wrapped


def threaded(fn, *args, **kwargs):
    callback = kwargs.pop("callback", None)
    resp = fn(*args, **kwargs)
    if callable is not None:
        callback(resp)


class Projector:
    def __init__(
        self, search_id: int, projector_id: int, projector=DEFAULT_PROJECTOR, **kwargs
    ):
        self.search_id = search_id
        self.projector_id = projector_id
        self.is_fitted = False
        self.is_fitting = False
        self.is_projected = False
        self.is_projecting = False
        self.projection = None
        self.classifications = None

        settings = {**DEFAULT_PROJECTOR_SETTINGS}
        for key, value in kwargs.items():
            if key in DEFAULT_PROJECTOR_SETTINGS and value is not None:
                settings[key] = value

        self.projector = projector(**settings)
        self.settings = settings

    def project(self, X: np.ndarray, callback=None):
        if not self.is_fitted:
            return None

        if self.projection is None:
            self.is_projecting = True
            try:
                _thread.start_new_thread(
                    threaded,
                    (self.projector.transform, X),
                    {"callback": projected(self, callback)},
                )
            finally:
                self.is_projecting = False

        return self.projection

    def fit(self, X: np.ndarray, y: np.ndarray = None, callback=None):
        self.is_fitted = False
        self.is_fitting = True
        try:
            _thread.start_new_thread(
                threaded,
                (self.projector.fit, X),
                {"y": y, "callback": fitted(self, callback)},
            )
        except Exception:
            self.is_fitted = False
            self.is_fitting = False

    def load(self, dumped_projector: bytes):
        with BytesIO(dumped_projector) as b:
            try:
                self.projector = joblib.load(b)
                self.is_fitted = True
            except (RuntimeError, EOFError):
                # Projector model seems to be broken.
                self.is_fitted = False

    def dump(self):
        with BytesIO() as b:
            joblib.dump(self.projector, b)
            return b.getvalue()
