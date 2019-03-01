import createPubSub from 'pub-sub-es';
import React from 'react';
import ReactDOM from 'react-dom';
import { ConnectedRouter } from 'react-router-redux';
import { Provider } from 'react-redux';
import { createState, history } from './factories/state';

// HOCs
import { Provider as PubSubProvider } from './hocs/pub-sub';

// Components
import App from './components/App';
import AppFake from './components/AppFake';

// Actions
import { setServerStartTime } from './actions';

// Utils
import getServer from './utils/get-server';
import Logger from './utils/logger';
import registerServiceWorker from './registerServiceWorker';

// Styles
import './index.scss';

const logger = Logger('Index');

// Initialize store
const state = createState();
let rehydratedStore;
const storeRehydrated = state.configure();

// Init pub-sub service
const pubSub = createPubSub();

const getServerStartTime = store =>
  fetch(`${getServer()}/api/v1/started/`)
    .then(response => response.text())
    .then(time => ({ store, serverStartTime: parseInt(time, 10) }));

const render = (Component, store, error) => {
  if (!store) {
    ReactDOM.render(
      <PubSubProvider value={pubSub}>
        <AppFake error={error} />
      </PubSubProvider>,
      document.getElementById('root')
    );
  } else {
    ReactDOM.render(
      <Provider store={store}>
        <ConnectedRouter history={history}>
          <PubSubProvider value={pubSub}>
            <Component />
          </PubSubProvider>
        </ConnectedRouter>
      </Provider>,
      document.getElementById('root')
    );
  }
};

render(AppFake);

storeRehydrated
  .then(getServerStartTime)
  .then(({ store, serverStartTime }) => {
    if (store.getState().present.serverStartTime !== serverStartTime) {
      // The server restarted, hence we need to reset the store as we don't
      // know whether the server has persistent data or not.
      state.reset();
      store.dispatch(setServerStartTime(serverStartTime));
    }
    rehydratedStore = store;
    render(App, store);
  })
  .catch(error => {
    logger.error('Failed to rehydrate the store! This is fatal!', error);
    render(
      undefined,
      undefined,
      'Failed to initialize! Did you start the server? Otherwise, please contact an admin.'
    );
  });

if (module.hot) {
  module.hot.accept('./components/App', () => {
    const NextApp = require('./components/App').default; // eslint-disable-line global-require
    render(NextApp, rehydratedStore);
  });
  storeRehydrated.then(store => {
    window.store = store;
  });
}

registerServiceWorker();
