import PropTypes from 'prop-types';
import React from 'react';
import { connect } from 'react-redux';

// Components
import ButtonIcon from '../components/ButtonIcon';
import SubTopBar from '../components/SubTopBar';
import SubTopBottomBarButtons from '../components/SubTopBottomBarButtons';
import ToolTip from '../components/ToolTip';

// Actions
import { setHiglassMouseTool, setShowAutoencodings } from '../actions';

// Utils
import { Logger } from '../utils';

// Configs
import { PAN_ZOOM, SELECT } from '../configs/mouse-tools';

// Stylesheets
import './HomeSubTopBar.scss';

const logger = Logger('ViewerSubTopBar'); // eslint-disable-line

const HomeSubTopBar = props => (
  <SubTopBar className="home-sub-top-bar" wrap={true}>
    <SubTopBottomBarButtons className="flex-c flex-a-c no-list-style">
      <li>
        <ToolTip
          align="left"
          delayIn={1000}
          delayOut={500}
          title={
            <span className="flex-c">
              <span>Pan & Zoom Tool</span>
              <span className="short-cut">Z</span>
            </span>
          }
        >
          <ButtonIcon
            icon="drag"
            iconOnly={true}
            isActive={props.mouseTool === PAN_ZOOM}
            onClick={() => props.setMouseTool(PAN_ZOOM)}
          />
        </ToolTip>
      </li>
      <li>
        <ToolTip
          align="left"
          delayIn={1000}
          delayOut={500}
          title={
            <span className="flex-c">
              <span>Select Tool</span>
              <span className="short-cut">S</span>
            </span>
          }
        >
          <ButtonIcon
            icon="select"
            iconOnly={true}
            isActive={props.mouseTool === SELECT}
            onClick={() => props.setMouseTool(SELECT)}
          />
        </ToolTip>
      </li>
      <li className="separator" />
      <li>
        <ToolTip
          align="left"
          delayIn={2000}
          delayOut={500}
          title={
            <span className="flex-c">
              <span>Show Autoencodings</span>
            </span>
          }
        >
          <ButtonIcon
            icon="autoencoding"
            iconOnly={true}
            isActive={props.showAutoencodings}
            onClick={() => {
              props.setShowAutoencodings(!props.showAutoencodings);
            }}
          />
        </ToolTip>
      </li>
    </SubTopBottomBarButtons>
    <SubTopBottomBarButtons className="flex-c flex-a-c flex-jc-e no-list-style">
      <li>
        <ToolTip
          align="right"
          delayIn={1000}
          delayOut={500}
          title={
            <span className="flex-c">
              {props.rangeSelection[0] === null ? (
                <span>Select a region to search</span>
              ) : (
                <span>Search for section</span>
              )}
              <span className="short-cut">X</span>
            </span>
          }
        >
          <ButtonIcon
            icon="magnifier"
            className={`${props.rangeSelection[0] !== null ? 'primary' : ''}`}
            onClick={() => props.search(props.rangeSelection)}
            isDisabled={props.rangeSelection[0] === null}
          >
            Search
          </ButtonIcon>
        </ToolTip>
      </li>
    </SubTopBottomBarButtons>
  </SubTopBar>
);

HomeSubTopBar.propTypes = {
  setMouseTool: PropTypes.func,
  viewConfig: PropTypes.object,
  mouseTool: PropTypes.string,
  rangeSelection: PropTypes.array,
  setShowAutoencodings: PropTypes.func,
  showAutoencodings: PropTypes.bool,
  search: PropTypes.func
};

const mapStateToProps = state => ({
  viewConfig: state.present.viewConfig,
  mouseTool: state.present.higlassMouseTool,
  showAutoencodings: state.present.showAutoencodings
});

const mapDispatchToProps = dispatch => ({
  setMouseTool: mouseTool => dispatch(setHiglassMouseTool(mouseTool)),
  setShowAutoencodings: showAutoencodings =>
    dispatch(setShowAutoencodings(showAutoencodings))
});

export default connect(mapStateToProps, mapDispatchToProps)(HomeSubTopBar);
