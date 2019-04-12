class InvalidConfig(Exception):
    """Raised when the config is improperly defined"""

    pass


class LabelsDidNotChange(Exception):
    """Raised when a classifier is tried to be trained on the same labels"""

    pass


class TooFewLabels(Exception):
    """Raised when a classifier is tried to be trained with too few labels"""

    pass
