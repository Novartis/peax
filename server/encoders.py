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


class Encoders:
    def __init__(self):
        self.encoders = []
        self.encoders_by_type = {}
        self.window_size = None
        self.resolution = None

        self._total_len_windows = 0
        self._total_len_encoded = 0

    def __iter__(self):
        return iter(self.encoders)

    @property
    def total_len_windows(self):
        return self._total_len_windows

    @property
    def total_len_encoded(self):
        return self._total_len_encoded

    def add(self, encoder):
        if self.window_size is None:
            self.window_size = encoder.window_size
        elif self.window_size != encoder.window_size:
            raise ValueError("Window sizes need to be the same")

        if self.resolution is None:
            self.resolution = encoder.resolution
        elif self.resolution != encoder.resolution:
            raise ValueError("Window sizes need to be the same")

        self.encoders.append(encoder)
        self.encoders_by_type[encoder.content_type] = encoder

        self._total_len_windows += int(self.window_size // encoder.resolution)
        self._total_len_encoded += encoder.latent_dim

    def get(self, dtype: str):
        if dtype in self.encoders_by_type:
            return self.encoders_by_type[dtype]
        raise KeyError("No encoder of type '{}' found".format(dtype))

    def export(self):
        return [encoder.export() for encoder in self.encoders]
