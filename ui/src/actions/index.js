import { ActionCreators } from 'redux-undo';

export const redo = ActionCreators.redo();

export const reset = () => ({
  type: 'RESET',
  payload: {}
});

export const setSearchId = searchId => ({
  type: 'SET_SEARCH_ID',
  payload: { searchId }
});

export const setServerStartTime = serverStartTime => ({
  type: 'SET_SERVER_START_TIME',
  payload: { serverStartTime }
});

export const setHomeInfoBarClose = homeInfoBarClose => ({
  type: 'SET_HOME_INFO_BAR_CLOSE',
  payload: { homeInfoBarClose }
});

export const setViewConfig = viewConfig => ({
  type: 'SET_VIEW_CONFIG',
  payload: { viewConfig }
});

export const setHiglassMouseTool = higlassMouseTool => ({
  type: 'SET_HIGLASS_MOUSE_TOOL',
  payload: { higlassMouseTool }
});

export const setSearchRightBarInfoMetadata = searchRightBarInfoMetadata => ({
  type: 'SET_SEARCH_RIGHT_BAR_INFO_METADATA',
  payload: { searchRightBarInfoMetadata }
});

export const setSearchRightBarProjectionSettings = searchRightBarProjectionSettings => ({
  type: 'SET_SEARCH_RIGHT_BAR_PROJECTION_SETTINGS',
  payload: { searchRightBarProjectionSettings }
});

export const setSearchRightBarShow = searchRightBarShow => ({
  type: 'SET_SEARCH_RIGHT_BAR_SHOW',
  payload: { searchRightBarShow }
});

export const setSearchRightBarTab = searchRightBarTab => ({
  type: 'SET_SEARCH_RIGHT_BAR_TAB',
  payload: { searchRightBarTab }
});

export const setSearchRightBarWidth = searchRightBarWidth => ({
  type: 'SET_SEARCH_RIGHT_BAR_WIDTH',
  payload: { searchRightBarWidth }
});

export const setSearchSelection = searchSelection => ({
  type: 'SET_SEARCH_SELECTION',
  payload: { searchSelection }
});

export const setSearchTab = searchTab => ({
  type: 'SET_SEARCH_TAB',
  payload: { searchTab }
});

export const setShowAutoencodings = showAutoencodings => ({
  type: 'SET_SHOW_AUTOENCODINGS',
  payload: { showAutoencodings }
});

export const undo = ActionCreators.undo();
