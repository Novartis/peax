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

import base64
import os
import cytoolz as toolz
import numpy as np
import sys
import time
from flask import Flask
from flask import request, jsonify, send_from_directory
from flask_cors import CORS
from hgtiles import cooler

from server import (
    bigwig,
    chromsizes,
    projector as projClazz,
    sampling,
    utils,
    vector,
    view_config,
)
from server.classifiers import Classifiers
from server.database import DB
from server.projectors import Projectors


def create(
    config,
    ext_filetype_handlers: list = None,
    clear_cache: bool = False,
    clear_db: bool = False,
    verbose: bool = False,
):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0" if verbose else "3"

    STARTED = int(time.time())

    # Init db
    db = DB(db_path=config.db_path, clear=clear_db)

    # Load autoencoders
    encoders = config.encoders
    datasets = config.datasets

    # Prepare data: load and encode windows
    datasets.prepare(encoders, config, clear=clear_cache, verbose=verbose)

    # Determine the absolute offset for windows
    abs_offset = np.inf
    abs_ends = 0
    abs_len = 0
    for chrom in config.chroms:
        abs_len += datasets.chromsizes[chrom]
        abs_offset = min(abs_offset, datasets.chromsizes_cum[chrom])
        abs_ends = max(
            abs_ends, datasets.chromsizes_cum[chrom] + datasets.chromsizes[chrom]
        )

    with datasets.cache() as dsc:
        # Load all the encodings into memory
        encodings = dsc.encodings[:]

        # Set up classifiers
        classifiers = Classifiers(
            db, encodings, window_size=encoders.window_size, abs_offset=abs_offset
        )

        # Set up projectors
        projectors = Projectors(db, encodings, encoders.window_size, abs_offset)

    app = Flask(__name__, static_url_path="", static_folder="../ui/build")
    CORS(app)

    ################
    # UI ENDPOINTS #
    ################

    @app.route("/")
    def view_root():
        return send_from_directory("ui/build", "index.html")

    @app.route("/<path:filename>")
    def view_root_files(filename):
        return send_from_directory("ui/build", filename)

    ####################
    # SEARCH ENDPOINTS #
    ####################

    @app.route("/api/v1/started/", methods=["GET"])
    def started():
        return str(STARTED)

    @app.route("/api/v1/info/", methods=["GET"])
    def info():
        info = {}
        min_win = np.inf
        max_win = 0

        for dataset in datasets:
            if dataset.content_type in encoders.encoders_by_type:
                encoder = encoders.encoders_by_type[dataset.content_type]
                info[dataset.id] = {
                    "windowSize": encoder.window_size,
                    "resolution": encoder.resolution,
                }
                min_win = min(min_win, encoder.window_size)
                max_win = max(max_win, encoder.window_size)

        info["windowSizeMin"] = min_win
        info["windowSizeMax"] = max_win
        info["minClassifications"] = config.min_classifications

        return jsonify(info)

    @app.route("/api/v1/search/", methods=["GET", "POST", "DELETE"])
    def search():
        if request.method == "GET":
            search_id = request.args.get("id")
            max_res = int(request.args.get("max", "-1"))
            info = db.get_search(search_id)

            if info is None and search_id is not None:
                return (
                    jsonify({"error": "Search #{} not found".format(search_id)}),
                    404,
                )

            if search_id is None:
                if max_res > 0:
                    info = info[:max_res]
                for i in info:
                    i["viewHeight"], i["maxViewHeight"] = view_config.height(
                        datasets, config
                    )
                    i["dataFrom"] = int(abs_offset)
                    i["dataTo"] = int(abs_ends)
                    i["windowSize"] = encoders.window_size
            else:
                info["viewHeight"], info["maxViewHeight"] = view_config.height(
                    datasets, config
                )
                info["dataFrom"] = int(abs_offset)
                info["dataTo"] = int(abs_ends)
                info["windowSize"] = encoders.window_size

            return jsonify(info)

        elif request.method == "POST":
            body = request.get_json()

            if body is None:
                return (jsonify({"error": "Did you forgot to send something? 😑"}), 400)

            window = body.get("window")

            if window is None:
                return (
                    jsonify({"error": "Search window needs to be specified! 😐"}),
                    400,
                )

            new_search = db.create_search(window, config)

            return jsonify({"info": "New search started", "id": new_search[0]})

        elif request.method == "DELETE":
            id = request.args.get("id")
            db.delete_search(id)
            return jsonify({"info": "It's all gone babe! Gone for good."})

        return jsonify({"error": "Unsupported action"}), 500

    @app.route("/api/v1/seeds/", methods=["GET"])
    def seeds():
        search_id = request.args.get("s")
        # allow_empty = request.args.get("allow-empty")

        if search_id is None:
            return (
                jsonify({"error": "Specify the search via the `s` URL parameter."}),
                400,
            )

        info = db.get_search(search_id)

        if info is None:
            return (
                jsonify({"error": "Unknown search with id '{}'".format(search_id)}),
                404,
            )

        # Get absolute locus and enforce it to be of the correct window size
        target_locus_abs = utils.enforce_window_size(
            info["target_from"], info["target_to"], encoders.window_size
        )

        target_locus_rel = target_locus_abs - abs_offset

        # Get chromosomal position
        target_locus_chrom = list(
            bigwig.abs2chr(
                datasets.chromsizes,
                target_locus_abs[0],
                target_locus_abs[1],
                is_idx2chr=True,
            )
        )

        if len(target_locus_chrom) > 1:
            return (
                jsonify({"error": "Search window is spanning chromosome border."}),
                400,
            )

        total_len = 0
        for encoder in encoders:
            total_len += encoder.latent_dim

        target = np.zeros(total_len)

        remove_windows = None

        pos = 0
        for dataset in datasets:
            encoder = encoders.get(dataset.content_type)
            step_size = encoders.window_size / config.step_freq

            window_from_idx = int(target_locus_rel[0] // step_size)
            window_from_start = int(window_from_idx * step_size)
            window_to_idx = window_from_idx + config.step_freq
            bins = int(encoders.window_size // encoder.resolution)
            offset = int(
                np.round((target_locus_rel[0] - window_from_start) / encoder.resolution)
            )

            target[pos : pos + encoder.latent_dim] = encoder.encode(
                bigwig.get(dataset.filepath, *target_locus_chrom[0], bins).reshape(
                    (1, bins, 1)
                )
            )

            if remove_windows is None:
                # Remove windows that overlap too much with the target search
                offset = (
                    target_locus_rel[0] - window_from_start
                ) / encoders.window_size
                max_offset = 0.66  # For which we remove the window
                k = np.ceil(config.step_freq * (offset - max_offset))
                remove_windows = np.arange(window_from_idx + k, window_to_idx + k)

            pos += encoder.latent_dim

        with datasets.cache() as dsc:
            num_windows = dsc.encodings.shape[0]

            # Array determining which data points should be used
            data_idx = np.ones(num_windows).astype(bool)

            # Remove the windows overlapping the target window (nt = no target)
            if np.max(remove_windows) >= 0 and np.min(remove_windows) < num_windows:
                data_idx[remove_windows.astype(int)] = False

            # Get classifications as already classified windows should be ignored
            classifications = np.array(
                list(
                    map(
                        lambda classif: int(classif["windowId"]),
                        db.get_classifications(search_id),
                    )
                )
            ).astype(int)

            # Remove already classified windows
            data_idx[classifications] = False

            classifier = classifiers.get(search_id)
            encodings = dsc.encodings[:]
            if classifier:
                _, p_y = classifier.predict(encodings)

            if classifications.size > 0 and classifier is not None:
                seeds = sampling.sample_by_uncertainty_density(
                    encodings, data_idx, target, p_y[:, 0]
                )

            elif classifications.size > 0 and classifier is None:
                # We need to train the classifier first
                classifier = classifiers.new(search_id)
                return jsonify(
                    {
                        "classifierId": classifier.classifier_id,
                        "isTrained": classifier.is_trained,
                        "isTraining": classifier.is_training,
                    }
                )

            else:
                # Remove empty windows (ne = no empty)
                data_idx[np.where((datasets.windows_max[:] < 0.1))] = False
                seeds = sampling.sample_by_dist_density(encodings, data_idx, target)

            assert np.unique(seeds).size == seeds.size, "Do not return duplicated seeds"

            return jsonify({"results": seeds.tolist()})

    @app.route("/api/v1/predictions/", methods=["GET"])
    def predictions():
        search_id = request.args.get("s")
        classifier_id = request.args.get("c")

        if search_id is None:
            return jsonify({"error": "Search id (`s`) is missing."}), 400

        classifier = classifiers.get(search_id, classifier_id)

        # Get search target window IDs
        search = db.get_search(search_id)
        search_target_windows = utils.get_target_window_idx(
            search["target_from"],
            search["target_to"],
            encoders.window_size,
            search["config"]["step_freq"],
            abs_offset,
        )

        with datasets.cache() as dsc:
            num_window = dsc.encodings.shape[0]
            fit_y, p_y = classifier.predict(dsc.encodings)

        window_ids = np.arange(num_window)

        # Exclude search target windows by setting their prediction to `0`
        if (
            np.min(search_target_windows[1]) >= 0
            and np.max(search_target_windows[1]) < num_window
        ):
            fit_y[np.arange(*search_target_windows[1]).astype(int)] = 0

        # Only regard positive classifications
        positive = np.where(fit_y == 1)
        window_ids_pos = window_ids[positive]
        p_y_pos = p_y[positive]

        sorted_idx = np.argsort(p_y_pos[:, 1])[::-1]
        window_ids_pos[sorted_idx]
        p_y_pos[sorted_idx]

        results = []

        # Get manual classifications
        classifications = db.get_classifications(search_id)
        classifications_hashed = utils.hashify(classifications, "windowId")

        probs_pos = p_y_pos[sorted_idx][:, 1].flatten()

        results_hashed = {}
        for i, window_id in enumerate(window_ids_pos[sorted_idx].tolist()):
            result = {
                "windowId": window_id,
                "probability": probs_pos[i],
                "classification": None,
            }

            if window_id in classifications_hashed:
                result["classification"] = classifications_hashed[window_id][
                    "classification"
                ]

            results_hashed[window_id] = len(results)
            results.append(result)

        for i, c in enumerate(classifications):
            if c["classification"] == 1 and c["windowId"] not in results_hashed:
                result = {
                    "windowId": window_id,
                    "probability": p_y[window_id],
                    "classification": None,
                }

        return jsonify({"results": results})

    @app.route("/api/v1/classes/", methods=["GET"])
    def view_classes():
        search_id = request.args.get("s")

        if search_id is None:
            return jsonify({"error": "Search id (`s`) is missing."}), 400

        # Manual classifications
        classifications = db.get_classifications(search_id)

        # Get search target window IDs
        search = db.get_search(search_id)
        search_target_windows = utils.get_target_window_idx(
            search["target_from"],
            search["target_to"],
            encoders.window_size,
            search["config"]["step_freq"],
            abs_offset,
        )

        with datasets.cache() as dsc:
            num_windows = dsc.windows.shape[0]

        classes = np.zeros(num_windows)

        # Manually classified regions
        for classification in classifications:
            clazz = 0
            if classification["classification"] == -1:
                clazz = 1
            if classification["classification"] == 1:
                clazz = 2

            classes[classification["windowId"]] = clazz

        # The search target
        classes[np.arange(*search_target_windows[1])] = 3

        return jsonify(
            {
                "results": base64.b64encode(classes.astype(np.uint8).tobytes()).decode(
                    "ascii"
                ),
                "encoding": "base64",
                "dtype": "uint8",
            }
        )

    @app.route("/api/v1/probabilities/", methods=["GET"])
    def view_probabilities():
        search_id = request.args.get("s")
        classifier_id = request.args.get("c")

        if search_id is None:
            return jsonify({"error": "Search id (`s`) is missing."}), 400

        classifier = classifiers.get(search_id, classifier_id)

        with datasets.cache() as dsc:
            num_windows = dsc.windows.shape[0]
            out = np.zeros(num_windows)

            if classifier is None:
                out[:] = 0.5
            else:
                # Load all encodings into memory. If this gets too slow or infeasible to
                # to compute we need to start using `warm_start`. See the following:
                # https://stackoverflow.com/a/30758348/981933
                fit_y, p_y = classifier.predict(dsc.encodings[:])
                out[:] = p_y[:, 1]

        return jsonify(
            {
                "results": base64.b64encode(out.astype(np.float32).tobytes()).decode(
                    "ascii"
                ),
                "encoding": "base64",
                "dtype": "float32",
            }
        )

    @app.route("/api/v1/classifier/", methods=["DELETE", "GET", "POST"])
    def view_classifier():
        search_id = request.args.get("s")
        classifier_id = request.args.get("c")
        if search_id is None:
            return jsonify({"error": "No search id (`s`) specified."}), 400

        if request.method == "DELETE":
            classifiers.delete(search_id, classifier_id)
            msg = " has" if classifier_id else "s have"
            return jsonify({"info": "Classifier{} been deleted.".format(msg)})

        elif request.method == "GET":
            clf = classifiers.get(search_id)

            if clf is None:
                return (
                    jsonify(
                        {
                            "error": "No classifier for search #{} found".format(
                                search_id
                            )
                        }
                    ),
                    404,
                )

            return jsonify(
                {
                    "classifierId": clf.classifier_id,
                    "featureImportance": clf.model.feature_importances_.tolist(),
                    "isTrained": clf.is_trained,
                    "isTraining": clf.is_training,
                }
            )

        elif request.method == "POST":
            # Compare classifications to last classifier
            classifier = classifiers.new(search_id)

            if classifier is None:
                return (jsonify({"error": "Classifications did not change"}), 409)

            return jsonify(
                {
                    "classifierId": classifier.classifier_id,
                    "isTrained": classifier.is_trained,
                    "isTraining": classifier.is_training,
                }
            )

        return jsonify({"error": "Unsupported action"}), 500

    @app.route("/api/v1/classifications/", methods=["GET"])
    def classifications():
        search_id = request.args.get("s")

        if search_id is None:
            return jsonify({"error": "Search id (`s`) is missing."}), 400

        classifications = db.get_classifications(search_id)

        return jsonify({"results": classifications})

    @app.route("/api/v1/classification/", methods=["GET", "PUT", "DELETE"])
    def classification():
        if request.method == "GET":
            search_id = request.args.get("sid")
            window_id = request.args.get("wid")

            if search_id is None:
                return (
                    jsonify(
                        {
                            "error": (
                                "Please tell us for which search you need the "
                                "classification(s) by specifying the `sid` URL "
                                "parameter."
                            )
                        }
                    ),
                    400,
                )

            classification = db.get_classification(search_id, window_id)

            return jsonify({"results": classification})

        elif request.method == "PUT":
            body = request.get_json()

            if body is None:
                return jsonify({"error": "Where's the payload? 🤨"}), 400

            search_id = body.get("searchId")
            window_id = body.get("windowId")
            classification = body.get("classification")

            if search_id is None or window_id is None or classification is None:
                return (
                    jsonify(
                        {
                            "error": (
                                "O Props, Where Art Thou? 🧐 Show em some love and "
                                "provide `searchId`, `windowId`, and `classification`."
                            )
                        }
                    ),
                    400,
                )

            if classification == "positive":
                classification = 1
            elif classification == "negative":
                classification = -1
            else:
                classification = 0

            db.set_classification(search_id, window_id, classification)

            return jsonify({"info": "Window was successfully classified."})

        elif request.method == "DELETE":
            body = request.get_json()

            if body is None:
                return jsonify({"error": "Where's the payload? 🤨"}), 400

            search_id = body.get("searchId")
            window_id = body.get("windowId")

            if search_id is None or window_id is None:
                return (
                    jsonify(
                        {
                            "error": (
                                "O Props, Where Art Thou? 🧐 Show em some love and "
                                "provide `searchId` and `windowId`."
                            )
                        }
                    ),
                    400,
                )

            db.delete_classification(search_id, window_id)

            return jsonify({"info": "Classification was successfully deleted."})

        return jsonify({"error": "Unsupported action"}), 500

    @app.route("/api/v1/data-tracks/", methods=["GET"])
    def view_data_tracks():
        return jsonify({"results": list(map(lambda x: x.id, datasets))})

    @app.route("/api/v1/view-height/", methods=["GET"])
    def view_configs_height():
        height = view_config.height(datasets, config)
        return jsonify({"height": height})

    @app.route("/api/v1/projection/", methods=["DELETE", "GET", "PUT"])
    def view_projection():
        search_id = request.args.get("s")
        projector_id = request.args.get("p")

        if search_id is None:
            return jsonify({"error": "Search ID (`s`) is missing"}), 400

        if request.method == "DELETE":
            projectors.delete(search_id, projector_id)
            msg = " has" if projector_id else "s have"
            return jsonify({"info": "Projection{} been deleted.".format(msg)})

        if request.method == "GET":
            projector = projectors.get(search_id, projector_id)

            if projector is None:
                return (
                    jsonify(
                        {
                            "error": "No projection for search #{} found".format(
                                search_id
                            )
                        }
                    ),
                    404,
                )

            with utils.catch(AttributeError) as projection:
                with datasets.cache() as dsc:
                    projection = base64.b64encode(
                        projector.project(dsc.encodings[:]).tobytes()
                    ).decode("ascii")

            # If the projector is already fitted the following call will do nothing
            projectors.fit(search_id, projector.projector_id)

            return jsonify(
                {
                    "projection": projection,
                    "projectionDtype": "float32",
                    "projectionEncoding": "base64",
                    "projectionIsProjecting": projector.is_projecting,
                    "projectorId": projector.projector_id,
                    "projectorIsFitted": projector.is_fitted,
                    "projectorIsFitting": projector.is_fitting,
                    "projectorSettings": projector.settings,
                }
            )

        elif request.method == "PUT":
            with utils.catch(
                ValueError,
                TypeError,
                default=projClazz.DEFAULT_PROJECTOR_SETTINGS["n_neighbors"],
            ) as n_neighbors:
                n_neighbors = int(request.args.get("nn"))

            with utils.catch(
                ValueError,
                TypeError,
                default=projClazz.DEFAULT_PROJECTOR_SETTINGS["min_dist"],
            ) as min_dist:
                min_dist = float(request.args.get("md"))

            projector = projectors.new(search_id, n_neighbors, min_dist)

            return jsonify(
                {
                    "projectorId": projector.projector_id,
                    "projectionIsProjecting": projector.is_projecting,
                    "projectorIsFitted": projector.is_fitted,
                    "projectorIsFitting": projector.is_fitting,
                }
            )

    #####################
    # HIGLASS ENDPOINTS #
    #####################

    @app.route("/version.txt", methods=["GET"])
    def version():
        return "SERVER_VERSION: 0.1.0-flask"

    @app.route("/api/v1/viewconfs/", methods=["GET"])
    def view_configs():
        view_id = request.args.get("d")

        if view_id == "default":
            return jsonify(view_config.build(datasets, config, default=True))

        if view_id == "default.e":
            return jsonify(
                view_config.build(
                    datasets, config, default=True, incl_autoencodings=True
                )
            )

        if view_id is None:
            infos = db.get_search()
            view_configs = {}
            for info in infos:
                view_configs[info["id"]] = view_config.build(datasets, config, info)
            return jsonify(view_configs)

        parts = view_id.split(".")

        with utils.catch(IndexError, default=None) as search_id:
            search_id = parts[0]

        with utils.catch(IndexError, default=None) as window_id:
            window_id = parts[1]

        with utils.catch(IndexError, default="") as options:
            options = parts[2]

        search_info = db.get_search(search_id)

        if utils.is_int(search_id, True):
            incl_predictions = search_info["classifiers"] > 0 and options.find("p") >= 0
            incl_autoencodings = options.find("e") >= 0

            if window_id is not None and utils.is_int(window_id, True):
                info = db.get_search(search_id)
                if info is not None:
                    step_size = encoders.window_size // config.step_freq
                    target_from_rel = step_size * int(window_id)
                    target_to_rel = target_from_rel + step_size
                    target_abs = list(
                        map(
                            int,
                            bigwig.chr2abs(
                                datasets.chromsizes,
                                config.chroms[0],  # First chrom defines the offset
                                target_from_rel,
                                target_to_rel,
                            ),
                        )
                    )
                    return jsonify(
                        view_config.build(
                            datasets,
                            config,
                            search_info=search_info,
                            domain=target_abs,
                            incl_predictions=incl_predictions,
                            incl_autoencodings=incl_autoencodings,
                            hide_label=True,
                        )
                    )

            else:
                info = db.get_search(search_id)
                if info is not None:
                    return jsonify(
                        view_config.build(
                            datasets,
                            config,
                            search_info=info,
                            incl_predictions=incl_predictions,
                            incl_autoencodings=incl_autoencodings,
                        )
                    )

        return (jsonify({"error": "Unknown view config with id: {}".format(id)}), 404)

    @app.route("/api/v1/available-chrom-sizes/", methods=["GET"])
    def available_chrom_sizes():
        return jsonify(
            {
                "count": len(chromsizes.all),
                "results": {i: chromsizes.all[i] for i in chromsizes.all},
            }
        )

    @app.route("/api/v1/chrom-sizes/", methods=["GET"])
    def chrom_sizes():
        id = request.args.get("id")
        res_type = request.args.get("type", "json")
        incl_cum = request.args.get("cum", False)

        if id is None:
            return jsonify(chromsizes.all)

        try:
            data = chromsizes.all[id]
        except KeyError:
            return jsonify({"error": "Not found"}), 404

        if incl_cum:
            cum = 0
            for chrom in data.keys():  # dictionaries in py3.6+ are ordered!
                data[chrom]["offset"] = cum
                cum += data[chrom]["size"]

        if res_type == "json":
            return jsonify(data)

        elif res_type == "csv":
            if incl_cum:
                return "\n".join(
                    "{}\t{}\t{}".format(chrom, row["size"], row["offset"])
                    for chrom, row in data.items()
                )
            else:
                return "\n".join(
                    "{}\t{}".format(chrom, row["size"]) for chrom, row in data.items()
                )

        else:
            return jsonify({"error": "Unknown response type"}), 500

    # NEW!
    @app.route("/api/v1/uids-by-filename/", methods=["GET"])
    def uids_by_filename():
        return jsonify(
            {
                "count": datasets.length,
                "results": {dataset.id: dataset.filename for dataset in datasets},
            }
        )

    @app.route("/api/v1/tilesets/", methods=["GET"])
    def tilesets():
        return jsonify(
            {
                "count": datasets.length,
                "next": None,
                "previous": None,
                "results": datasets.export(use_uuid=True),
            }
        )

    @app.route("/api/v1/tileset_info/", methods=["GET"])
    def tileset_info():
        uuids = request.args.getlist("d")

        search_res = db.get_search()

        # Get searches for special tilesets
        searches = list(
            map(
                lambda x: {
                    "uuid": "s{}p".format(x["id"]),
                    "search_id": x["id"],
                    "filetype": "__prediction__",
                },
                search_res,
            )
        )

        autoencodings = datasets.export(use_uuid=True, autoencodings=True)

        dataset_search_defs = datasets.export(use_uuid=True) + searches + autoencodings

        info = {}
        for uuid in uuids:
            ts = next((ts for ts in dataset_search_defs if ts["uuid"] == uuid), None)

            if ts is not None:
                info[uuid] = ts.copy()

                # see if there's a filepath provided
                filetype = info[uuid].get("filetype")
                filepath = info[uuid].get("filepath")

                if ext_filetype_handlers and filetype in ext_filetype_handlers:
                    handler = ext_filetype_handlers[filetype]["tileset_info"]
                    if filepath is not None:
                        info[uuid].update(handler(filepath))
                    else:
                        info[uuid].update(handler())
                elif bigwig.is_bigwig(filepath, filetype):
                    info[uuid] = {**bigwig.TILESET_INFO, **info[uuid]}
                    info[uuid].update(bigwig.tileset_info(ts["filepath"]))
                elif filetype == "cooler":
                    info[uuid].update(cooler.tileset_info(ts["filepath"]))
                elif filetype == "__autoencoding__":
                    info[uuid] = {**vector.TILESET_INFO, **info[uuid]}
                    info[uuid].update(
                        vector.tileset_info(datasets.chromsizes, encoders.resolution)
                    )
                elif filetype == "__prediction__":
                    info[uuid] = {**vector.TILESET_INFO, **info[uuid]}
                    info[uuid].update(
                        vector.tileset_info(
                            datasets.chromsizes, encoders.window_size / config.step_freq
                        )
                    )
                else:
                    print(
                        "Unknown filetype:", info[uuid].get("filetype"), file=sys.stderr
                    )
            else:
                info[uuid] = {"error": "No such tileset with uid: {}".format(uuid)}

        return jsonify(info)

    @app.route("/api/v1/tiles/", methods=["GET"])
    def tiles():
        tids_requested = set(request.args.getlist("d"))

        if not tids_requested:
            return jsonify({"error": "No tiles requested"}), 400

        def extract_uuid(tid):
            return tid.split(".")[0]

        uuids_to_tids = toolz.groupby(extract_uuid, tids_requested)

        search_res = db.get_search()

        # Get searches for special tilesets
        searches = list(
            map(
                lambda x: {
                    "uuid": "s{}p".format(x["id"]),
                    "search_id": x["id"],
                    "filetype": "__prediction__",
                },
                search_res,
            )
        )

        autoencodings = datasets.export(use_uuid=True, autoencodings=True)

        dataset_search_defs = datasets.export(use_uuid=True) + searches + autoencodings

        tiles = []
        for uuid, tids in uuids_to_tids.items():
            ts = next((ts for ts in dataset_search_defs if ts["uuid"] == uuid), None)
            if ts is not None:
                filetype = ts.get("filetype")
                filepath = ts.get("filepath")

                if ext_filetype_handlers and filetype in ext_filetype_handlers:
                    handler = ext_filetype_handlers[filetype]["tiles"]
                    if filepath is not None:
                        tiles.extend(handler(filepath, tids))
                    else:
                        tiles.extend(handler(tids))
                elif bigwig.is_bigwig(filepath, filetype):
                    tiles.extend(bigwig.tiles(filepath, tids))
                elif filetype == "cooler":
                    tiles.extend(cooler.tiles(filepath, tids))
                elif filetype == "__autoencoding__":
                    dataset = datasets.get(uuid.split("|")[0])
                    with dataset.cache() as dsc:
                        tiles.extend(
                            vector.tiles(
                                dsc.autoencodings,
                                encoders.resolution,
                                abs_len,
                                abs_offset,
                                tids,
                                datasets.chromsizes,
                            )
                        )
                elif filetype == "__prediction__":
                    classifier = classifiers.get(ts["search_id"])

                    if classifier is None:
                        return jsonify({})

                    with datasets.cache() as dsc:
                        _, p_y = classifier.predict(dsc.encodings[:])

                    p_y_merged = utils.merge_interleaved(
                        p_y[:, 1], config.step_freq, aggregator=np.nanmax
                    )

                    res_merged = int(encoders.window_size / config.step_freq)

                    tiles.extend(
                        vector.tiles(
                            p_y_merged,
                            res_merged,
                            abs_len,  # Absolute length of the chromosome
                            abs_offset,
                            tids,
                            datasets.chromsizes,
                            aggregator=np.max,
                            scaleup_aggregator=np.median,
                        )
                    )
                else:
                    print("Unknown filetype:", filetype, file=sys.stderr)

        data = {tid: tval for tid, tval in tiles}
        return jsonify(data)

    return app
