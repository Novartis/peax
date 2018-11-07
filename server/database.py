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

import json
import sqlite3
from server.defaults import DB_PATH


def objectify_search(search):
    updated = max(search[5] or "", search[6] or "", search[8] or "")

    return {
        "id": search[0],
        "target_from": search[1],
        "target_to": search[2],
        "config": json.loads(search[3]),
        "classifiers": search[9] if search[9] is not None else 0,
        "created": search[4],
        "updated": updated,
        "classifications": search[7] if search[7] is not None else 0,
    }


def objectify_classification(classif):
    return {
        "windowId": classif[1],
        "classification": classif[2],
        "created": classif[3],
        "updated": classif[4],
    }


def objectify_classifier(classifier):
    if classifier is None:
        return None

    return {
        "search_id": classifier[0],
        "classifier_id": classifier[1],
        "serialized_classifications": classifier[2],
        "model": classifier[3],
        "created": classifier[4],
        "updated": classifier[5],
    }


def objectify_projector(projector):
    if projector is None:
        return None

    return {
        "search_id": projector[0],
        "projector_id": projector[1],
        "projector": projector[2],
        "projection": projector[3],
        "classifications": projector[4],
        "settings": projector[5],
        "created": projector[6],
        "updated": projector[7],
    }


class DB:
    def __init__(self, db_path=DB_PATH, clear=False):
        self.db_path = db_path
        self.create_tables(clear=clear)

    def connect(self):
        return sqlite3.connect(self.db_path)

    def create_tables(self, clear=False):
        conn = self.connect()
        if clear:
            conn.execute("DROP TABLE IF EXISTS search")
            conn.execute("DROP TABLE IF EXISTS classification")
            conn.execute("DROP TABLE IF EXISTS classifier")
            conn.execute("DROP TABLE IF EXISTS projector")
            conn.execute("DROP TRIGGER IF EXISTS SearchUpdated")
            conn.execute("DROP TRIGGER IF EXISTS ClassificationUpdated")
            conn.execute("DROP TRIGGER IF EXISTS ClassifierUpdated")
            conn.execute("DROP TRIGGER IF EXISTS ProjectorUpdated")
            conn.commit()

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS search
            (
                id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                target_from INT,
                target_to INT,
                config TEXT,
                created DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
                updated DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
                name TEXT,
                description TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS SearchUpdated
                AFTER UPDATE ON search FOR EACH ROW
                BEGIN
                    UPDATE search
                    SET updated = CURRENT_TIMESTAMP
                    WHERE id = old.id;
                END
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS classification
            (
                search_id INT NOT NULL,
                window_id INT NOT NULL,
                is_positive INT,
                created DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (search_id) REFERENCES search(id),
                PRIMARY KEY (search_id, window_id)
            )
            """
        )

        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS ClassificationUpdated
                AFTER UPDATE ON classification FOR EACH ROW
                BEGIN
                    UPDATE classification
                    SET updated = CURRENT_TIMESTAMP
                    WHERE
                        search_id = old.search_id
                        AND window_id = old.window_id;
                END
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS classifier
            (
                search_id INT NOT NULL,
                classifier_id INT NOT NULL,
                serialized_classifications BLOB,
                model BLOB,
                created DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (search_id) REFERENCES search(id),
                PRIMARY KEY (search_id, classifier_id)
            )
            """
        )

        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS ClassifierUpdated
                AFTER UPDATE ON classifier FOR EACH ROW
                BEGIN
                    UPDATE classifier
                    SET updated = CURRENT_TIMESTAMP
                    WHERE
                        search_id = old.search_id
                        AND classifier_id = old.classifier_id;
                END
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projector
            (
                search_id INT NOT NULL,
                projector_id INT NOT NULL,
                projector BLOB,
                projection BLOB,
                classifications BLOB,
                settings TEXT,
                created DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (search_id) REFERENCES search(id),
                PRIMARY KEY (search_id, projector_id)
            )
            """
        )

        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS ProjectorUpdated
                AFTER UPDATE ON projector FOR EACH ROW
                BEGIN
                    UPDATE projector
                    SET updated = CURRENT_TIMESTAMP
                    WHERE
                        search_id = old.search_id
                        AND projector_id = old.projector_id;
                END
            """
        )

        conn.commit()
        conn.close()

    def create_search(self, target, config):
        with self.connect() as conn:
            # We need the curser as the connection doesn't feature `lastrowid`
            c = conn.cursor()

            str_config = json.dumps(config.export(), sort_keys=True)
            c.execute(
                "INSERT INTO search (target_from, target_to, config) "
                "VALUES (?, ?, ?)",
                (target[0], target[1], str_config),
            )
            id = c.lastrowid
            conn.commit()

            return c.execute("SELECT * FROM search WHERE id = ?", (id,)).fetchone()

    def get_search(self, id=None):
        results = []

        with self.connect() as conn:
            if id is not None:
                result = conn.execute(
                    """
                    SELECT
                        s.*,
                        c.updated,
                        c.classifications,
                        x.updated,
                        x.classifiers
                    FROM
                        search AS s
                        LEFT OUTER JOIN (
                            SELECT
                                search_id,
                                MAX(updated) AS updated,
                                COUNT(*) AS classifications
                            FROM classification
                            WHERE search_id = ?
                            GROUP BY search_id
                        ) AS c
                        ON s.id == c.search_id
                        LEFT OUTER JOIN (
                            SELECT
                                search_id,
                                MAX(updated) AS updated,
                                COUNT(*) AS classifiers
                            FROM classifier
                            WHERE search_id = ?
                            GROUP BY search_id
                        ) AS x
                        ON s.id == x.search_id
                    WHERE
                        s.id = ?
                    """,
                    (id, id, id),
                ).fetchone()

                if result is None:
                    return None

                return objectify_search(result)

            results = list(
                map(
                    objectify_search,
                    conn.execute(
                        """
                    SELECT
                        s.*,
                        c.updated,
                        c.classifications,
                        x.updated,
                        x.classifiers
                    FROM
                        search AS s
                        LEFT OUTER JOIN (
                            SELECT
                                search_id,
                                MAX(updated) AS updated,
                                COUNT(*) AS classifications
                            FROM classification
                            GROUP BY search_id
                        ) AS c
                        ON s.id == c.search_id
                        LEFT OUTER JOIN (
                            SELECT
                                search_id,
                                MAX(updated) AS updated,
                                COUNT(*) AS classifiers
                            FROM classifier
                            GROUP BY search_id
                        ) AS x
                        ON s.id == x.search_id
                    GROUP BY
                        s.id
                    ORDER BY
                        MAX(
                            s.updated,
                            COALESCE(c.updated, 0),
                            COALESCE(x.updated, 0)
                        )
                        DESC
                """
                    ).fetchall(),
                )
            )

        return results

    def delete_search(self, id):
        with self.connect() as conn:
            conn.execute("DELETE FROM search WHERE id = ?", (id,))

    def get_classification(self, search_id, window_id=None):
        with self.connect() as conn:
            if window_id is None:
                return list(
                    map(
                        objectify_classification,
                        conn.execute(
                            "SELECT * FROM classification WHERE search_id = ?",
                            (search_id,),
                        ).fetchall(),
                    )
                )

            return objectify_classification(
                conn.execute(
                    """
                SELECT *
                FROM classification
                WHERE
                    search_id = ? AND window_id = ?
                """,
                    (search_id, window_id),
                ).fetchone()
            )

    def get_classifications(self, search_id):
        return self.get_classification(search_id)

    def set_classification(self, search_id, window_id, is_positive):
        with self.connect() as conn:
            conn.execute(
                """
                    UPDATE
                        classification
                    SET
                        is_positive = ?
                    WHERE
                        search_id = ? AND window_id = ?
                """,
                (is_positive, search_id, window_id),
            )
            conn.execute(
                """
                    INSERT OR IGNORE INTO
                        classification(search_id, window_id, is_positive)
                    VALUES
                        (?, ?, ?);
                """,
                (search_id, window_id, is_positive),
            )
            conn.commit()

    def delete_classification(self, search_id, window_id):
        with self.connect() as conn:
            return conn.execute(
                """
                DELETE FROM classification
                WHERE
                    search_id = ? AND window_id = ?
                """,
                (search_id, window_id),
            )

    def create_classifier(self, search_id, classif: bytes = b""):
        with self.connect() as conn:
            classifier_id = conn.execute(
                """
                SELECT MAX(classifier_id)
                FROM classifier
                WHERE search_id = ?
                """,
                (search_id,),
            ).fetchone()[0]

            classifier_id = classifier_id + 1 if classifier_id is not None else 0

            conn.execute(
                """
                INSERT INTO
                    classifier
                    (search_id, classifier_id, serialized_classifications)
                VALUES
                    (?, ?, ?)
                """,
                (search_id, classifier_id, classif),
            )

            conn.commit()

            return classifier_id

    def get_classifier(self, search_id: int, classifier_id: int = None):
        with self.connect() as conn:
            if classifier_id is not None:
                return objectify_classifier(
                    conn.execute(
                        """
                        SELECT *
                        FROM classifier
                        WHERE search_id = ? AND classifier_id = ?
                        """,
                        (search_id, classifier_id),
                    ).fetchone()
                )

            return objectify_classifier(
                conn.execute(
                    """
                    SELECT *
                    FROM classifier
                    WHERE search_id = ?
                    ORDER BY classifier_id DESC
                    """,
                    (search_id,),
                ).fetchone()
            )

    def set_classifier(self, search_id, classifier_id, dumped_model):
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE classifier
                SET model = ?
                WHERE search_id = ? and classifier_id = ?
                """,
                (dumped_model, search_id, classifier_id),
            )
            conn.commit()

    def delete_classifier(self, search_id: int, classifier_id: int = None):
        with self.connect() as conn:
            if classifier_id is not None:
                return conn.execute(
                    """
                    DELETE FROM classifier
                    WHERE search_id = ? AND classifier_id = ?
                    """,
                    (search_id, classifier_id),
                )

            return conn.execute(
                """
                DELETE FROM classifier
                WHERE search_id = ?
                """,
                (search_id,),
            )

    def create_projector(
        self,
        search_id,
        projector: bytes = b"",
        projection: bytes = b"",
        classifications: bytes = b"",
        settings: bytes = b"",
    ):
        with self.connect() as conn:
            projector_id = conn.execute(
                """
                SELECT MAX(projector_id)
                FROM projector
                WHERE search_id = ?
                """,
                (search_id,),
            ).fetchone()[0]

            projector_id = projector_id + 1 if projector_id is not None else 0

            conn.execute(
                """
                INSERT INTO
                    projector
                    (search_id, projector_id, projector, projection, classifications, settings)
                VALUES
                    (?, ?, ?, ?, ?, ?)
                """,
                (
                    search_id,
                    projector_id,
                    projector,
                    projection,
                    classifications,
                    settings,
                ),
            )

            conn.commit()

            return projector_id

    def get_projector(self, search_id: int, projector_id: int = None):
        with self.connect() as conn:
            if projector_id is not None:
                return objectify_projector(
                    conn.execute(
                        """
                        SELECT *
                        FROM projector
                        WHERE search_id = ? AND projector_id = ?
                        """,
                        (search_id, projector_id),
                    ).fetchone()
                )

            return objectify_projector(
                conn.execute(
                    """
                    SELECT *
                    FROM projector
                    WHERE search_id = ?
                    ORDER BY projector_id DESC
                    """,
                    (search_id,),
                ).fetchone()
            )

    def set_projector(
        self,
        search_id: int,
        projector_id: int,
        projector: bytes = b"",
        projection: bytes = b"",
        classifications: bytes = b"",
        settings: bytes = b"",
    ):
        with self.connect() as conn:
            if len(projector):
                conn.execute(
                    """
                    UPDATE projector
                    SET projector = ?
                    WHERE search_id = ? and projector_id = ?
                    """,
                    (projector, search_id, projector_id),
                )

            if len(projection):
                conn.execute(
                    """
                    UPDATE projector
                    SET projection = ?
                    WHERE search_id = ? and projector_id = ?
                    """,
                    (projection, search_id, projector_id),
                )

            if len(classifications):
                conn.execute(
                    """
                    UPDATE projector
                    SET classifications = ?
                    WHERE search_id = ? and projector_id = ?
                    """,
                    (classifications, search_id, projector_id),
                )

            if len(settings):
                conn.execute(
                    """
                    UPDATE projector
                    SET settings = ?
                    WHERE search_id = ? and projector_id = ?
                    """,
                    (settings, search_id, projector_id),
                )

            conn.commit()

    def delete_projector(self, search_id: int, projector_id: int = None):
        with self.connect() as conn:
            if projector_id is not None:
                return conn.execute(
                    """
                    DELETE FROM projector
                    WHERE search_id = ? AND projector_id = ?
                    """,
                    (search_id, projector_id),
                )

            return conn.execute(
                """
                DELETE FROM projector
                WHERE search_id = ?
                """,
                (search_id,),
            )
