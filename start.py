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


class MyParser(argparse.ArgumentParser):

    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)


parser = MyParser(description="Peak Explorer CLI")
parser.add_argument("-e", "--encoder", help="path to saved encoder")
parser.add_argument("-d", "--dataset", help="path to saved dataset (bigwig)")
parser.add_argument(
    "-w", "--windowsize", help="path to saved dataset (bigwig)"
)
parser.add_argument("-r", "--resolution", help="number of bp per bin")
parser.add_argument(
    "-s",
    "--stepsize",
    help="relative to window, e.g., `2` => `windowsize / 2 = stepsize in bp`",
)
parser.add_argument(
    "-c", "--chroms", help="comma-separated list of chromosomes to search over"
)
parser.add_argument("--config", help="use config file instead of args")
parser.add_argument(
    "--clear", action="store_true", help="clears the db on startup"
)
parser.add_argument("--debug", action="store_true", help="debug flag")
parser.add_argument(
    "--host", help="Customize the hostname", default="localhost"
)
parser.add_argument("--port", help="Customize the port", default=5000)
parser.add_argument("--verbose", action="store_true", help="verbose flag")

args = parser.parse_args()

if args.config:
    with open(args.config, "r") as f:
        config = json.load(f)

    app = server.create(
        config["aes"],
        config["datasets"],
        config["config"],
        db_path=config.get("db_path", None),
        clear=args.clear,
        verbose=args.verbose,
    )
else:
    app = server.create(
        args.encoder,
        args.dataset,
        args.windowsize,
        args.resolution,
        args.stepsize,
        args.chroms,
        args.verbose,
    )

app.run(debug=args.debug, host=args.host, port=args.port)
