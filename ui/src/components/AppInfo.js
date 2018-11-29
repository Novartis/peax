import React from 'react';
// Utils
import getServer from '../utils/get-server';
import Logger from '../utils/logger';

const URL = `${getServer()}/version.txt`;

const logger = Logger('AppInfo');

class AppInfo extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      serverVersion: <em>(loading&hellip;)</em>
    };

    fetch(URL)
      .then(response => response.text())
      .then(response => {
        if (response.split('\n')[0].substr(0, 14) !== 'SERVER_VERSION') {
          throw Error(
            'Could not parse `version.txt`. Expecting the first line to start with `SERVER_VERSION`.'
          );
        }

        this.setState({
          serverVersion: response.split('\n')[0].slice(16)
        });
      })
      .catch(error => {
        logger.warn('Could not retrieve or parse server version.', error);
        this.setState({
          serverVersion: <em>(unknown)</em>
        });
      });
  }

  render() {
    return (
      <div className="app-info">
        <ul className="no-list-style">
          <li>
            <strong>Peax</strong>: Version {VERSION_PEAX}
          </li>
          <li>
            <strong>HiGlass</strong>: Version {VERSION_HIGLASS}
          </li>
          <li>
            <strong>Server</strong>: Version {this.state.serverVersion}
          </li>
        </ul>
      </div>
    );
  }
}

export default AppInfo;
