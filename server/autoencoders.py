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

from server.autoencoder import Autoencoder


class Autoencoders:

    def __init__(self, ae_defs: list) -> list:
        self.aes = []
        self.aes_by_type = {}
        self.window_size = None
        self.resolution = None
        for ae_def in ae_defs:
            ae = Autoencoder(ae_def)
            self.aes.append(ae)
            self.aes_by_type[ae_def["content_type"]] = ae

            if self.window_size is None:
                self.window_size = ae_def["window_size"]
            elif self.window_size != ae_def["window_size"]:
                raise ValueError("Window sizes need to be the same")

            if self.resolution is None:
                self.resolution = ae_def["resolution"]
            elif self.resolution != ae_def["resolution"]:
                raise ValueError("Window sizes need to be the same")

    def get(self, etype: str):
        if etype in self.aes_by_type:
            return self.aes_by_type[etype]
        return None
