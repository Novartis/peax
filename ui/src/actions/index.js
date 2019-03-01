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

export const setSearchHover = searchHover => ({
  type: 'SET_SEARCH_HOVER',
  payload: { searchHover }
});

export const setSearchRightBarMetadata = searchRightBarMetadata => ({
  type: 'SET_SEARCH_RIGHT_BAR_METADATA',
  payload: { searchRightBarMetadata }
});

export const setSearchRightBarProgress = searchRightBarProgress => ({
  type: 'SET_SEARCH_RIGHT_BAR_PROGRESS',
  payload: { searchRightBarProgress }
});

export const setSearchRightBarHelp = searchRightBarInfoHelp => ({
  type: 'SET_SEARCH_RIGHT_BAR_HELP',
  payload: { searchRightBarInfoHelp }
});

export const setSearchRightBarProjection = searchRightBarProjection => ({
  type: 'SET_SEARCH_RIGHT_BAR_PROJECTION',
  payload: { searchRightBarProjection }
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
