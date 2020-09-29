import json
import os
import pandas as pd
import pathlib
import sys

from collections import OrderedDict
from typing import Dict, List, TypeVar

from server.chromsizes import all as all_chromsizes, SUPPORTED_CHROMOSOMES
from server.dataset import Dataset
from server.datasets import Datasets
from server.defaults import CLASSIFIER, CLASSIFIER_PARAMS, CACHE_DIR, CACHING, COORDS, DB_PATH, STEP_FREQ, MIN_CLASSIFICATIONS
from server.encoder import Autoencoder, Encoder
from server.encoders import Encoders
from server.exceptions import InvalidConfig

# Any list-like object needs to have the *same* variable type. I.e., `List[Num]` does
# not allow [1, "chr1"]. It must either contain ints only or str only.
IntOrStr = TypeVar('Num', int, str)


class Config:
    def __init__(self, config_file: Dict, base_data_dir: str = None):
        # Init
        self.encoders = Encoders()
        self.datasets = Datasets()

        if base_data_dir is None:
            self.base_data_dir = os.getcwd()
        else:
            self.base_data_dir = base_data_dir

        # For custom encoder models
        module_path = os.path.abspath(os.path.join(self.base_data_dir))
        if module_path not in sys.path:
            sys.path.append(module_path)

        # Helper
        self._default_chroms = True

        # Set defaults
        self.classifier = CLASSIFIER
        self.classifier_params = CLASSIFIER_PARAMS
        self.coords = COORDS
        self.step_freq = STEP_FREQ
        self.min_classifications = MIN_CLASSIFICATIONS
        self.db_path = DB_PATH
        self.cache_dir = CACHE_DIR
        self.caching = CACHING
        self.variable_target = False
        self.normalize_tracks = False

        self._chroms = SUPPORTED_CHROMOSOMES[self.coords]
        self._chromsizes = None
        self._chromsizes = None
        self._custom_chromosomes = None

        # Set file
        self.file = config_file

    def config(self, config_file):
        keys = set(config_file.keys())

        # Custom chromsizes need to be set prior to other properties
        if "chromsizes" in keys:
            self.set("chromsizes", config_file["chromsizes"])

            # If we have custom chromsizes we likely have custom chroms too
            if "chroms" in keys:
                self.set("chroms", config_file["chroms"])

        for key in keys:
            if key != "encoders" and key != "datasets" and key != "chromsizes":
                self.set(key, config_file[key])

        encoders = config_file["encoders"]

        if isinstance(config_file["encoders"], str):
            try:
                with open(os.path.join(self.base_data_dir, config_file["encoders"]), "r") as f:
                    encoders = json.load(f).values()
            except FileNotFoundError:
                print(
                    "You specified that the encoder config is provided in another "
                    "file that does not exist. Make sure that `encoders` points "
                    "to a valid encoder definition file."
                )
                raise

        for encoder in encoders:
            if 'from_file' in encoder:
                try:
                    with open(os.path.join(self.base_data_dir, encoder['from_file']), "r") as f:
                        encoder_config = json.load(f)[encoder['content_type']]
                except FileNotFoundError:
                    print(
                        "You specified that the encoder config is provided in another "
                        "file that does not exist. Make sure that `from_file` points "
                        "to a valid encoder definition file."
                    )
                    raise
                except KeyError:
                    print(
                        "No predefined encoder of type {} found".format(encoder['content_type'])
                    )
                    raise

                for key in encoder_config:
                    encoder.setdefault(key, encoder_config[key])

            model_args = encoder.get("model_args", [])
            for i, model_arg in enumerate(model_args):
                if isinstance(model_arg, str):
                    model_args[i] = model_arg.format(base_data_dir=self.base_data_dir)

            try:
                self.add(
                    Autoencoder(
                        autoencoder_filepath=os.path.join(self.base_data_dir, encoder["autoencoder"]),
                        content_type=encoder["content_type"],
                        window_size=encoder["window_size"],
                        resolution=encoder["resolution"],
                        channels=encoder["channels"],
                        input_dim=encoder["input_dim"],
                        latent_dim=encoder["latent_dim"],
                        model_args=model_args,
                    )
                )
            except KeyError:
                try:
                    self.add(
                        Autoencoder(
                            encoder_filepath=os.path.join(self.base_data_dir, encoder["encoder"]),
                            decoder_filepath=os.path.join(self.base_data_dir, encoder["decoder"]),
                            content_type=encoder["content_type"],
                            window_size=encoder["window_size"],
                            resolution=encoder["resolution"],
                            channels=encoder["channels"],
                            input_dim=encoder["input_dim"],
                            latent_dim=encoder["latent_dim"],
                            model_args=model_args,
                        )
                    )
                except KeyError:
                    self.add(
                        Encoder(
                            encoder_filepath=os.path.join(self.base_data_dir, encoder["encoder"]),
                            content_type=encoder["content_type"],
                            window_size=encoder["window_size"],
                            resolution=encoder["resolution"],
                            channels=encoder["channels"],
                            input_dim=encoder["input_dim"],
                            latent_dim=encoder["latent_dim"],
                            model_args=model_args,
                        )
                    )

        for ds in config_file["datasets"]:
            self.add(
                Dataset(
                    filepath=os.path.join(self.base_data_dir, ds["filepath"]),
                    content_type=ds.get("content_type", "unknown"),
                    id=ds["id"],
                    name=ds["name"],
                    coords=self.coords,
                    chromsizes=self.chromsizes,
                    custom_chromosomes=self.custom_chromosomes,
                )
            )

    def add(self, o):
        if hasattr(o, "encode") and callable(getattr(o, "encode")):
            self.addEncoder(o)
            return

        if hasattr(o, "prepare") and callable(getattr(o, "prepare")):
            self.addDataset(o)
            return

        raise AttributeError("Unknown object type")

    def addEncoder(self, ae):
        self.encoders.add(ae)

    def addDataset(self, ds):
        self.datasets.add(ds)

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, value: Dict):
        if value.get("encoders") and value.get("datasets"):
            self._file = value
            if self._file:
                self.config(self._file)
        else:
            raise InvalidConfig("Config file needs to include `encoders` and `datasets`")

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, value: str):
        if value in all_chromsizes or self.chromsizes is not None:
            self._coords = value
            if self._default_chroms:
                self._chroms = SUPPORTED_CHROMOSOMES[value]
        else:
            raise InvalidConfig("Unknown coordinate system")

    @property
    def chroms(self):
        return self._chroms

    @chroms.setter
    def chroms(self, value: List[IntOrStr]):
        # fmt: off
        if (
            all(isinstance(v, str) for v in value) or
            all(isinstance(v, int) for v in value)
        ):
        # fmt: on
            self._chroms = value
            self._default_chroms = False
        else:
            raise InvalidConfig("Chromosomes must be a list of strings or ints")

    @property
    def chromsizes(self):
        return self._chromsizes

    @chromsizes.setter
    def chromsizes(self, value):
        try:
            sizes = OrderedDict()
            for chrom, size in value:
                sizes[chrom] = size
            self._chromsizes = pd.Series(sizes)
            self._custom_chromosomes = self._chromsizes.index.values.tolist()
        except:
            raise InvalidConfig("Chromsizes must be a list of string-int pairs")

    @property
    def custom_chromosomes(self):
        return self._custom_chromosomes

    @property
    def step_freq(self):
        return self._step_freq

    @step_freq.setter
    def step_freq(self, value: int):
        if value > 0:
            self._step_freq = value
        else:
            raise InvalidConfig("Step frequency must be larger than zero")

    @property
    def variable_target(self):
        return self._variable_target

    @variable_target.setter
    def variable_target(self, value: bool):
        self._variable_target = bool(value)

    @property
    def min_classifications(self):
        return self._min_classifications

    @min_classifications.setter
    def min_classifications(self, value: int):
        if value > 0:
            self._min_classifications = value
        else:
            raise InvalidConfig("Minimum classifications must be larger than zero")

    @property
    def db_path(self):
        return self._db_path

    @db_path.setter
    def db_path(self, value: str):
        if isinstance(value, str):
            self._db_path = os.path.join(self.base_data_dir, value)
            pathlib.Path(os.path.dirname(self._db_path)).mkdir(parents=True, exist_ok=True)
        else:
            raise InvalidConfig("Path to the database needs to be a string")

    @property
    def cache_dir(self):
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value: str):
        self._cache_dir = os.path.join(self.base_data_dir, value)
        pathlib.Path(self._cache_dir).mkdir(parents=True, exist_ok=True)

    @property
    def caching(self):
        return self._caching

    @caching.setter
    def caching(self, value: bool):
        self._caching = bool(value)

    def set(self, key, value):
        if key == "chroms":
            self.chroms = value

        elif key == "chromsizes":
            self.chromsizes = value

        elif key == "coords":
            self.coords = value

        elif key == "step_freq":
            self.step_freq = value

        elif key == "classifier":
            self.classifier = value

        elif key == "classifier_params":
            self.classifier_params = value

        elif key == "min_classifications":
            self.min_classifications = value

        elif key == "db_path":
            self.db_path = value

        elif key == "caching":
            self.caching = value

        elif key == "variable_target":
            self.variable_target = value

        elif key == "normalize_tracks":
            self.normalize_tracks = value

        else:
            raise InvalidConfig("Unknown settings: {}".format(key))

    def export(self, ignore_chromsizes: bool = False):
        return {
            "encoders": self.encoders.export(),
            "datasets": self.datasets.export(ignore_chromsizes=ignore_chromsizes),
            "chroms": self.chroms,
            "step_freq": self.step_freq,
            "min_classifications": self.min_classifications,
            "db_path": self.db_path,
        }
