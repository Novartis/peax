#!/usr/bin/env python

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
import argparse
import atexit
import json
import os
import sys


parser = argparse.ArgumentParser(description="Peak Explorer CLI")
parser.add_argument(
    "-c", "--config", help="path to your JSON config file", default="config.json"
)
parser.add_argument(
    "-b", "--base-data-dir",
    help="base directory which the config file refers to",
    default=None
)
parser.add_argument(
    "--clear", action="store_true", help="clears the cache and database on startup"
)
parser.add_argument(
    "--clear-cache", action="store_true", help="clears the cache on startup"
)
parser.add_argument(
    "--clear-cache-at-exit", action="store_true", help="clear the cache on shutdown"
)
parser.add_argument(
    "--clear-db", action="store_true", help="clears the database on startup"
)
parser.add_argument("-d", "--debug", action="store_true", help="turn on debug mode")
parser.add_argument("--host", help="customize the hostname", default="localhost")
parser.add_argument("--port", help="customize the port", default=5000)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="turn verbose logging on"
)

try:
    args = parser.parse_args()
except SystemExit as err:
    if err.code == 0:
        sys.exit(0)
    if err.code == 2:
        parser.print_help()
        sys.exit(0)
    raise

from server import server
from server.config import Config

try:
    with open(args.config, "r") as f:
        config_file = json.load(f)
except FileNotFoundError:
    print(
        "You need to either provide a config file via `--config` or "
        "have it as `config.json` in the root directory of Peax"
    )
    raise

verbose = args.verbose or args.debug

# Create a config object
config = Config(config_file, args.base_data_dir)

clear_cache = args.clear or args.clear_cache
clear_db = args.clear or args.clear_db

# Turn off clearing as Werkzeug is calling this script the second time
if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    clear_cache = False
    clear_db = False

# Create app instance
app = server.create(config, clear_cache=clear_cache, clear_db=clear_db, verbose=verbose)


def remove_cache(config):
    config.datasets.remove_cache()


if args.clear_cache_at_exit:
    atexit.register(remove_cache, config)

# Run the instance
app.run(debug=args.debug, host=args.host, port=args.port)
