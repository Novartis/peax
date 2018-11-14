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
import json
import sys

from server import server
from server.config import Config


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)


parser = MyParser(description="Peak Explorer CLI")
parser.add_argument("--config", help="use config file instead of args")
parser.add_argument("--clear", action="store_true", help="clears the db on startup")
parser.add_argument("--debug", action="store_true", help="debug flag")
parser.add_argument("--host", help="Customize the hostname", default="localhost")
parser.add_argument("--port", help="Customize the port", default=5000)
parser.add_argument("--verbose", action="store_true", help="verbose flag")

args = parser.parse_args()

config_path = args.config if args.config else "config.json"

try:
    with open(config_path, "r") as f:
        config_file = json.load(f)
except FileNotFoundError:
    print(
        "You need to provide either provide a config file via `--config` or "
        "have it as `config.json` in the root directory of Peax"
    )
    raise

# Create a config object
config = Config(config_file)

# Create app instance
app = server.create(config, clear=args.clear, verbose=args.verbose)

# Run the instance
app.run(debug=args.debug, host=args.host, port=args.port)
