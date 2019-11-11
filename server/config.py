import json
import pandas as pd
import pathlib
from collections import OrderedDict
from typing import Dict, List, TypeVar

from server.chromsizes import all as all_chromsizes
from server.dataset import Dataset
from server.datasets import Datasets
from server.defaults import CACHE_DIR, CACHING, CHROMS, COORDS, DB_PATH, STEP_FREQ, MIN_CLASSIFICATIONS
from server.encoder import Autoencoder, Encoder
from server.encoders import Encoders
from server.exceptions import InvalidConfig

# Any list-like object needs to have the *same* variable type. I.e., `List[Num]` does
# not allow [1, "chr1"]. It must either contain ints only or str only.
IntOrStr = TypeVar('Num', int, str)


class Config:
    def __init__(self, config_file: Dict):
        # Init
        self.encoders = Encoders()
        self.datasets = Datasets()

        # Set defaults
        self.coords = COORDS
        self.chroms = CHROMS
        self.step_freq = STEP_FREQ
        self.min_classifications = MIN_CLASSIFICATIONS
        self.db_path = DB_PATH
        self.cache_dir = CACHE_DIR
        self.caching = CACHING
        self.variable_target = False
        self.normalize_tracks = False

        # Set file
        self.file = config_file

    def config(self, config_file):
        keys = set(config_file.keys())

        # Custom chromsizes need to be set prior to other proerties
        if "chromsizes" in keys:
            self.set("chromsizes", config_file["chromsizes"])

        for key in keys:
            if key != "encoders" and key != "datasets" and key != "chromsizes":
                self.set(key, config_file[key])

        for encoder in config_file["encoders"]:
            if 'from_file' in encoder:
                try:
                    with open(encoder['from_file'], "r") as f:
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

            try:
                self.add(
                    Autoencoder(
                        autoencoder_filepath=encoder["autoencoder"],
                        content_type=encoder["content_type"],
                        window_size=encoder["window_size"],
                        resolution=encoder["resolution"],
                        channels=encoder["channels"],
                        input_dim=encoder["input_dim"],
                        latent_dim=encoder["latent_dim"],
                    )
                )
            except KeyError:
                try:
                    self.add(
                        Autoencoder(
                            encoder_filepath=encoder["encoder"],
                            decoder_filepath=encoder["decoder"],
                            content_type=encoder["content_type"],
                            window_size=encoder["window_size"],
                            resolution=encoder["resolution"],
                            channels=encoder["channels"],
                            input_dim=encoder["input_dim"],
                            latent_dim=encoder["latent_dim"],
                        )
                    )
                except KeyError:
                    self.add(
                        Encoder(
                            encoder_filepath=encoder["encoder"],
                            content_type=encoder["content_type"],
                            window_size=encoder["window_size"],
                            resolution=encoder["resolution"],
                            channels=encoder["channels"],
                            input_dim=encoder["input_dim"],
                            latent_dim=encoder["latent_dim"],
                        )
                    )

        for ds in config_file["datasets"]:
            self.add(
                Dataset(
                    filepath=ds["filepath"],
                    content_type=ds["content_type"],
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
        if value in all_chromsizes or self.chromsizes:
            self._coords = value
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
        else:
            raise InvalidConfig("Chromosomes must be a list of strings or ints")

    @property
    def chromsizes(self):
        if self._coords in all_chromsizes:
            return all_chromsizes[self._coords]
        return None

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
            self._db_path = value
        else:
            raise InvalidConfig("Path to the database needs to be a string")

    @property
    def cache_dir(self):
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value: str):
        pathlib.Path(value).mkdir(parents=True, exist_ok=True)
        self._cache_dir = value

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
