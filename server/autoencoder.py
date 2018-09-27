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

import keras
import numpy as np


class Autoencoder:

    def __init__(self, definition: dict):
        self.de_path = definition["decoder"]
        self.en_path = definition["encoder"]
        self.content_type = definition["content_type"]
        self.window_size = definition["window_size"]
        self.resolution = definition["resolution"]
        self.channels = definition["channels"]
        self.input_dim = definition["input_dim"]
        self.latent_dim = definition["latent_dim"]
        self.en = keras.models.load_model(self.en_path)
        self.de = keras.models.load_model(self.de_path)

    def autoencode(self, data: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(data))

    def decode(self, data: np.ndarray) -> np.ndarray:
        return self.de.predict(data)

    def encode(self, data: np.ndarray) -> np.ndarray:
        return self.en.predict(data)
