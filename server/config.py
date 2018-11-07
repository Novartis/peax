from server.dataset import Dataset
from server.datasets import Datasets
from server.defaults import CHROMS, DB_PATH, STEP_FREQ, MIN_CLASSIFICATIONS
from server.encoder import Autoencoder, Encoder
from server.encoders import Encoders
from server.exceptions import InvalidConfig


class Config:
    def __init__(self, config_file):
        self.encoders = Encoders()
        self.datasets = Datasets()
        self.chroms = CHROMS
        self.step_freq = STEP_FREQ
        self.min_classifications = MIN_CLASSIFICATIONS
        self.db_path = DB_PATH
        self.file = config_file

        if self.file:
            self.config(self.file)

    def config(self, config_file):
        keys = set(config_file.keys())
        for encoder in config_file["encoders"]:
            try:
                self.add(
                    Autoencoder(
                        encoder=encoder["encoder"],
                        decoder=encoder["decoder"],
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
                        encoder=encoder["encoder"],
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
                )
            )

        # Remove `encoders` and `datasets` keys so we can iterate over the rest
        keys.remove("encoders")
        keys.remove("datasets")

        for key in keys:
            self.set(key, config_file[key])

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

    def set(self, key, value):
        if key == "chroms":
            # fmt: off
            if (
                all(isinstance(v, str) for v in value) or
                all(isinstance(v, int) for v in value)
            ):
            # fmt: on
                self.chroms = value
            else:
                raise InvalidConfig("Chromosomes must be a list of strings or ints")
            return

        if key == "step_freq":
            if value > 0:
                self.step_freq = value
            else:
                raise InvalidConfig("Step frequency must be larger than zero")
            return

        if key == "min_classifications":
            if value > 0:
                self.min_classifications = value
            else:
                raise InvalidConfig("Minimum classifications must be larger than zero")
            return

        if key == "db_path":
            if isinstance(value, str):
                self.db_path = value
            else:
                raise InvalidConfig("Path to the database needs to be a string")
            return

        raise InvalidConfig("Unknown settings: {}".format(key))

    def export(self):
        return {
            "encoders": self.encoders.export(),
            "datasets": self.datasets.export(),
            "chroms": self.chroms,
            "step_freq": self.step_freq,
            "min_classifications": self.min_classifications,
            "db_path": self.db_path,
        }
