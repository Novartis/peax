import _thread


def done(instance, prefix: str, callback: callable = None):
    def wrapped():
        setattr(instance, "{}ed".format(prefix), True)
        setattr(instance, "{}ing".format(prefix), False)
        if callback is not None:
            callback()

    return wrapped


def update(instance, classifiers, outdated_classifier_ids, force: bool = False):
    _, outdated_classifier_ids = instance.outdated()
    if force:
        classifiers.evaluate_all(instance.search_id, update=force)
    else:
        for classifier_id in outdated_classifier_ids:
            classifiers.evaluate(instance.search_id, classifier_id, no_threading=True)
    instance.is_computed = True
    instance.is_computed = False


class Progress:
    def __init__(
        self,
        search_id: int,
        classifier_ids,
        unpredictability,
        uncertainty,
        convergence,
        divergence,
        num_labels,
    ):
        self.search_id = search_id

        self.classifier_ids = classifier_ids
        self.unpredictability = unpredictability
        self.uncertainty = uncertainty
        self.convergence = convergence
        self.divergence = divergence
        self.num_labels = num_labels

        is_outdated, outdated_classifier_ids = self.outdated()

        self.is_computed = not is_outdated
        self.is_computing = False

        self.outdated_classifier_ids = outdated_classifier_ids

    def update(self, classifiers, force: bool = False):
        if self.is_computing:
            return

        self.is_computed = False
        self.is_computing = True
        try:
            _thread.start_new_thread(update, (self, classifiers, force))
        except Exception:
            self.is_computed = False
            self.is_computing = False

    def outdated(self):
        if len(self.unpredictability) == 0:
            return False, []

        is_outdated = False
        outdated = []
        for i in range(len(self.unpredictability)):
            if self.unpredictability[i] is None:
                is_outdated = True
                outdated.append(i)

        return is_outdated, outdated

    def to_dict(self, camel_case: bool = False):
        out = {}

        unpredictability = None
        uncertainty = None
        convergence = None
        divergence = None

        if self.is_computed:
            unpredictability = self.unpredictability
            uncertainty = self.uncertainty
            convergence = self.convergence
            divergence = self.divergence

        if camel_case:
            out["searchId"] = self.search_id
            out["isComputed"] = self.is_computed
            out["isComputing"] = self.is_computing
        else:
            out["search_id"] = self.search_id
            out["is_computed"] = self.is_computed
            out["is_computing"] = self.is_computing

        out["unpredictability"] = unpredictability
        out["uncertainty"] = uncertainty
        out["convergence"] = convergence
        out["divergence"] = divergence

        return out
