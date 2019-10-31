import * as history from 'history';
import { routerMiddleware } from 'react-router-redux';
import { applyMiddleware, compose, createStore } from 'redux';
import { enableBatching } from 'redux-batched-actions';
import freeze from 'redux-freeze';
import { createLogger } from 'redux-logger';
import { autoRehydrate, persistStore, purgeStoredState } from 'redux-persist';
import { asyncSessionStorage } from 'redux-persist/storages';
import thunk from 'redux-thunk';
import undoable, { ActionCreators, groupByActionTypes } from 'redux-undo';

// Reducer
import rootReducer from '../reducers';

// Actions
import {
  reset as resetState,
  setViewConfig,
  setHiglassMouseTool,
  setSearchRightBarShow,
  setSearchRightBarWidth
} from '../actions';

const prefix = 'HiGlassApp.';

const config = {
  storage: asyncSessionStorage,
  debounce: 1000,
  keyPrefix: prefix
};

const browserHistory = history.createBrowserHistory();

const middleware = [
  autoRehydrate(),
  applyMiddleware(thunk),
  applyMiddleware(routerMiddleware(browserHistory))
];

if (process.env.NODE_ENV === 'development') {
  // Configure the logger middleware
  const logger = createLogger({
    level: 'info',
    collapsed: true
  });

  middleware.push(applyMiddleware(freeze));
  middleware.push(applyMiddleware(logger));
}

const configureStore = initialState => {
  const store = createStore(
    undoable(enableBatching(rootReducer), {
      groupBy: groupByActionTypes([
        setViewConfig().type,
        setHiglassMouseTool().type,
        setSearchRightBarShow().type,
        setSearchRightBarWidth().type
      ]),
      limit: 20
    }),
    initialState,
    compose(...middleware)
  );

  // Snippet to allow hot reload to work with reducers
  if (module.hot) {
    module.hot.accept(() => {
      store.replaceReducer(rootReducer);
    });
  }

  return new Promise((resolve, reject) => {
    persistStore(store, config, error => {
      if (error) {
        reject(error);
      } else {
        resolve(store);
      }
    });
  });
};

const createState = () => {
  let store;

  const configure = initialState =>
    configureStore(initialState).then(configuredStore => {
      store = configuredStore;
      return store;
    });

  const reset = () => {
    // Reset store
    store.dispatch(resetState());

    // Clear history
    store.dispatch(ActionCreators.clearHistory());

    // Purge persistent store
    return purgeStoredState(config);
  };

  return {
    get store() {
      return store;
    },
    configure,
    reset
  };
};

export { configureStore, browserHistory as history, createState };
