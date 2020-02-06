import PropTypes from 'prop-types';
import React from 'react';
import { connect } from 'react-redux';

// Actions
import {
  setSearchRightBarShow,
  setSearchRightBarTab,
  setSearchRightBarWidth
} from '../actions';

// Components
import Button from '../components/Button';
import RightBar from '../components/RightBar';
import RightBarSubTopBar from '../components/RightBarSubTopBar';
import RightBarContent from '../components/RightBarContent';
import TabContent from '../components/TabContent';
import TabTrigger from '../components/TabTrigger';
import SearchRightBarHelp from './SearchRightBarHelp';
import SearchRightBarInfo from './SearchRightBarInfo';

// Configs
import {
  RIGHT_BAR_MIN_WIDTH,
  TAB_RIGHT_BAR_INFO,
  TAB_RIGHT_BAR_HELP
} from '../configs/search';

const rightBarWidthToggler = props => () => {
  props.setRightBarShow(!props.rightBarShow);
};

const SearchRightBar = props => (
  <RightBar
    className="search-right-bar"
    isShown={props.rightBarShow}
    show={props.setRightBarShow}
    toggle={rightBarWidthToggler(props)}
    width={props.rightBarWidth}
    widthSetter={props.setRightBarWidth}
    widthSetterFinal={props.widthSetterFinal}
  >
    <RightBarSubTopBar>
      <TabTrigger
        for={TAB_RIGHT_BAR_INFO}
        tabChange={props.setRightBarTab}
        tabOpen={props.rightBarTab}
      >
        <Button>Info</Button>
      </TabTrigger>
      <TabTrigger
        for={TAB_RIGHT_BAR_HELP}
        tabChange={props.setRightBarTab}
        tabOpen={props.rightBarTab}
      >
        <Button>Help</Button>
      </TabTrigger>
    </RightBarSubTopBar>
    {props.rightBarShow && (
      <RightBarContent>
        <TabContent
          className="full-dim flex-c flex-v"
          for={TAB_RIGHT_BAR_INFO}
          tabOpen={props.rightBarTab}
        >
          <SearchRightBarInfo
            isComputingProgress={props.isComputingProgress}
            isErrorProgress={props.isErrorProgress}
            progress={props.progress}
            pubSub={props.pubSub}
            searchInfo={props.searchInfo}
          />
        </TabContent>
        <TabContent
          className="full-dim flex-c flex-v"
          for={TAB_RIGHT_BAR_HELP}
          tabOpen={props.rightBarTab}
        >
          <SearchRightBarHelp />
        </TabContent>
      </RightBarContent>
    )}
  </RightBar>
);

SearchRightBar.propTypes = {
  isComputingProgress: PropTypes.bool.isRequired,
  isErrorProgress: PropTypes.bool.isRequired,
  progress: PropTypes.object.isRequired,
  pubSub: PropTypes.object.isRequired,
  rightBarShow: PropTypes.bool.isRequired,
  rightBarTab: PropTypes.oneOfType([PropTypes.string, PropTypes.symbol])
    .isRequired,
  rightBarWidth: PropTypes.number.isRequired,
  searchInfo: PropTypes.object.isRequired,
  setRightBarShow: PropTypes.func.isRequired,
  setRightBarTab: PropTypes.func.isRequired,
  setRightBarWidth: PropTypes.func.isRequired,
  widthSetterFinal: PropTypes.func.isRequired
};

const mapStateToProps = state => ({
  rightBarShow: state.present.searchRightBarShow,
  rightBarTab: state.present.searchRightBarTab,
  rightBarWidth: state.present.searchRightBarWidth
});

const mapDispatchToProps = dispatch => ({
  setRightBarShow: rightBarShow =>
    dispatch(setSearchRightBarShow(rightBarShow)),
  setRightBarTab: rightBarTab => dispatch(setSearchRightBarTab(rightBarTab)),
  setRightBarWidth: rightBarWidth => {
    dispatch(
      setSearchRightBarWidth(Math.max(rightBarWidth, RIGHT_BAR_MIN_WIDTH))
    );
  }
});

export default connect(mapStateToProps, mapDispatchToProps)(SearchRightBar);
