from server import utils
from server.progress import Progress


class Progresses:
    def __init__(self, db, classifiers):
        self.db = db
        self.classifiers = classifiers
        self.progresses = {}

    def get(self, search_id: int, update: bool = False):
        progress_data = self.db.get_progress(search_id)

        classifier_ids = []
        unpredictability_all = []
        unpredictability_labels = []
        prediction_proba_change_all = []
        prediction_proba_change_labels = []
        convergence_all = []
        convergence_labels = []
        divergence_all = []
        divergence_labels = []
        num_labels = []

        for p in range(len(progress_data)):
            classifier_ids.append(progress_data[p][0])
            unpredictability_all.append(progress_data[p][1])
            unpredictability_labels.append(progress_data[p][2])
            prediction_proba_change_all.append(progress_data[p][3])
            prediction_proba_change_labels.append(progress_data[p][4])
            convergence_all.append(progress_data[p][5])
            convergence_labels.append(progress_data[p][6])
            divergence_all.append(progress_data[p][7])
            divergence_labels.append(progress_data[p][8])
            num_labels.append(utils.unserialize_classif(progress_data[p][9]).shape[0])

        try:
            progress = self.progresses[search_id]
        except KeyError:
            progress = Progress(
                search_id=search_id,
                classifier_ids=classifier_ids,
                unpredictability_all=unpredictability_all,
                unpredictability_labels=unpredictability_labels,
                prediction_proba_change_all=prediction_proba_change_all,
                prediction_proba_change_labels=prediction_proba_change_labels,
                convergence_all=convergence_all,
                convergence_labels=convergence_labels,
                divergence_all=divergence_all,
                divergence_labels=divergence_labels,
                num_labels=num_labels,
            )

        if len(classifier_ids) and (not progress.is_computed or update):
            progress.update(self.classifiers, force=update)

        return progress
