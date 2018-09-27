import { combineReducers } from 'redux';
import { routerReducer as routing } from 'react-router-redux';

import defaultSetReducer from '../utils/default-set-reducer';
import {
  TAB_RIGHT_BAR_PROJECTION,
  TAB_RESULTS,
} from '../configs/search';

const searchId = defaultSetReducer(
  'searchId', null
);

const serverStartTime = defaultSetReducer(
  'serverStartTime', -1
);

const homeInfoBarClose = defaultSetReducer(
  'homeInfoBarClose', false
);

const viewConfig = defaultSetReducer(
  'viewConfig', null
);

const higlassMouseTool = defaultSetReducer(
  'higlassMouseTool', 'panZoom'
);

const searchRightBarInfoMetadata = defaultSetReducer(
  'searchRightBarInfoMetadata', true
);

const searchRightBarProjectionSettings = defaultSetReducer(
  'searchRightBarProjectionSettings', false
);

const searchRightBarShow = defaultSetReducer(
  'searchRightBarShow', false
);

const searchRightBarTab = defaultSetReducer(
  'searchRightBarTab', TAB_RIGHT_BAR_PROJECTION
);

const searchRightBarWidth = defaultSetReducer(
  'searchRightBarWidth', 200
);

const searchRightBarProjectionCamera = defaultSetReducer(
  'searchRightBarProjectionCamera', [0, 0, 1]
);

const searchSelection = defaultSetReducer(
  'searchSelection', []
);

const searchTab = defaultSetReducer(
  'searchTab', TAB_RESULTS
);

const showAutoencodings = defaultSetReducer(
  'showAutoencodings', false
);

const appReducer = combineReducers({
  routing,
  searchId,
  serverStartTime,
  homeInfoBarClose,
  viewConfig,
  higlassMouseTool,
  searchRightBarInfoMetadata,
  searchRightBarProjectionSettings,
  searchRightBarShow,
  searchRightBarTab,
  searchRightBarWidth,
  searchRightBarProjectionCamera,
  searchSelection,
  searchTab,
  showAutoencodings,
});

const rootReducer = (state, action) => {
  if (action.type === 'RESET') {
    state = undefined;  // eslint-disable-line no-param-reassign
  }

  return appReducer(state, action);
};

export default rootReducer;
