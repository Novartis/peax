import PropTypes from "prop-types";
import React from "react";
import { connect } from "react-redux";

// Actions
import {
  setSearchRightBarShow,
  setSearchRightBarTab,
  setSearchRightBarWidth
} from "../actions";

// Components
import Button from "../components/Button";
import RightBar from "../components/RightBar";
import RightBarSubTopBar from "../components/RightBarSubTopBar";
import RightBarContent from "../components/RightBarContent";
import TabContent from "../components/TabContent";
import TabTrigger from "../components/TabTrigger";
import SearchRightBarInfo from "./SearchRightBarInfo";
import SearchRightBarProjection from "./SearchRightBarProjection";

// Configs
import {
  TAB_RIGHT_BAR_PROJECTION,
  TAB_RIGHT_BAR_INFO
} from "../configs/search";

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
        for={TAB_RIGHT_BAR_PROJECTION}
        tabChange={props.setRightBarTab}
        tabOpen={props.rightBarTab}
      >
        <Button>Projection</Button>
      </TabTrigger>
      <TabTrigger
        for={TAB_RIGHT_BAR_INFO}
        tabChange={props.setRightBarTab}
        tabOpen={props.rightBarTab}
      >
        <Button>Info</Button>
      </TabTrigger>
    </RightBarSubTopBar>
    {props.rightBarShow && (
      <RightBarContent>
        <TabContent
          className="full-dim flex-c flex-v"
          for={TAB_RIGHT_BAR_PROJECTION}
          tabOpen={props.rightBarTab}
        >
          <SearchRightBarProjection searchInfo={props.searchInfo} />
        </TabContent>
        <TabContent
          className="full-dim flex-c flex-v"
          for={TAB_RIGHT_BAR_INFO}
          tabOpen={props.rightBarTab}
        >
          <SearchRightBarInfo searchInfo={props.searchInfo} />
        </TabContent>
      </RightBarContent>
    )}
  </RightBar>
);

SearchRightBar.propTypes = {
  rightBarShow: PropTypes.bool,
  rightBarTab: PropTypes.oneOfType([PropTypes.string, PropTypes.symbol]),
  rightBarWidth: PropTypes.number,
  searchInfo: PropTypes.object,
  setRightBarShow: PropTypes.func,
  setRightBarTab: PropTypes.func,
  setRightBarWidth: PropTypes.func,
  widthSetterFinal: PropTypes.func
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
  setRightBarWidth: rightBarWidth =>
    dispatch(setSearchRightBarWidth(rightBarWidth))
});

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(SearchRightBar);
