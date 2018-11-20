/* eslint-env: node */

/** Copyright 2018 Novartis Institutes for BioMedical Research Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.* */

const fs = require("fs");
const changeCase = require("change-case");
const globalEnvironment = require("../config/gEnv").env;

const run = prod => {
  const config = {};

  try {
    const configBase = require("../config.json"); // eslint-disable-line global-require, import/no-unresolved
    Object.assign(config, configBase);
  } catch (ex) {
    // Nothing
  }

  try {
    const configLocal = require(`../config.${prod ? "prod" : "dev"}.json`); // eslint-disable-line global-require, import/no-unresolved, import/no-dynamic-require
    Object.assign(config, configLocal);
  } catch (e) {
    /* Nothing */
  }
  try {
    const configLocal = require("../config.local.json"); // eslint-disable-line global-require, import/no-unresolved
    Object.assign(config, configLocal);
  } catch (e) {
    /* Nothing */
  }
  const env = Object.keys(config)
    .filter(key => globalEnvironment.indexOf(key) >= 0)
    .map(
      key =>
        `window.HGAC_${changeCase.constantCase(key)}=${
          typeof key === "string" ? JSON.stringify(config[key]) : config[key]
        };`
    );
  fs.writeFile("./build/config.js", env.join("\n"), err => {
    if (err) {
      console.log(err);
    }
  });
};
module.exports = {
  run
};
