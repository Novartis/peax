from server import utils
from server.progress import Progress


class Progresses:
    def __init__(self, db, classifiers):
        self.db = db
        self.classifiers = classifiers
        self.progresses = {}

    def get(self, search_id: int, update: bool = False):
        classifier_info = self.db.get_classifier(search_id)

        if classifier_info is None:
            return

        progress_data = self.db.get_progress(search_id)

        classifier_ids = []
        unpredictability = []
        uncertainty = []
        convergence = []
        divergence = []
        num_labels = []
        for p in range(len(progress_data)):
            classifier_ids.append(progress_data[p][0])
            unpredictability.append(progress_data[p][1])
            uncertainty.append(progress_data[p][2])
            convergence.append(progress_data[p][3])
            divergence.append(progress_data[p][4])
            num_labels.append(utils.unserialize_classif(progress_data[p][5]).shape[0])

        try:
            progress = self.progresses[search_id]
        except KeyError:
            progress = Progress(
                search_id=search_id,
                classifier_ids=classifier_ids,
                unpredictability=unpredictability,
                uncertainty=uncertainty,
                convergence=convergence,
                divergence=divergence,
                num_labels=num_labels,
            )

        if not progress.is_computed:
            progress.update(self.classifiers)

        return progress
