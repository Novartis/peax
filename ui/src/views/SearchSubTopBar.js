import PropTypes from 'prop-types';
import React from 'react';
import { connect } from 'react-redux';

// Higher-order components
import { withPubSub } from '../hocs/pub-sub';

// Components
import AppInfo from '../components/AppInfo';
import ButtonIcon from '../components/ButtonIcon';
import SubTopBar from '../components/SubTopBar';
import SubTopBottomBarButtons from '../components/SubTopBottomBarButtons';
import ToolTip from '../components/ToolTip';

// Services
import { setShowAutoencodings } from '../actions';

// Utils
import { Deferred, Logger } from '../utils';

const logger = Logger('SearchSubTopBar');

const showInfo = pubSub => () => {
  pubSub.publish('globalDialog', {
    message: <AppInfo />,
    request: new Deferred(),
    resolveOnly: true,
    resolveText: 'Close',
    icon: 'logo',
    headline: 'Peax'
  });
};

const SearchSubTopBar = props => (
  <SubTopBar>
    <SubTopBottomBarButtons className="flex-c flex-a-c no-list-style">
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
      <li>
        <ToolTip
          align="left"
          delayIn={2000}
          delayOut={500}
          title={
            <span className="flex-c">
              <span>Normalize to this</span>
            </span>
          }
        >
          <ButtonIcon
            icon="ratio"
            iconOnly={true}
            isIconMirrorOnFocus={true}
            isActive={props.isMinMaxValuesByTarget}
            onClick={props.normalize}
          />
        </ToolTip>
      </li>
      <li>
        <ToolTip
          align="left"
          delayIn={2000}
          delayOut={500}
          title={
            <span className="flex-c">
              <span>Reset viewports</span>
              <span className="short-cut">R</span>
            </span>
          }
        >
          <ButtonIcon
            icon="reset"
            iconOnly={true}
            isDisabled={!props.viewportChanged}
            isIconRotationOnFocus={true}
            onClick={props.resetViewport}
          />
        </ToolTip>
      </li>
    </SubTopBottomBarButtons>
    <SubTopBottomBarButtons className="flex-c flex-a-c flex-jc-e no-list-style">
      <li>
        <ToolTip
          align="right"
          delayIn={2000}
          delayOut={500}
          title={
            <span className="flex-c">
              <span>Show App Information</span>
            </span>
          }
        >
          <ButtonIcon
            icon="info"
            iconOnly={true}
            onClick={showInfo(props.pubSub)}
          />
        </ToolTip>
      </li>
      <li>
        <ToolTip
          align="right"
          delayIn={2000}
          delayOut={500}
          title={
            <span className="flex-c">
              <span>Download Classification</span>
              <span className="short-cut">CMD + S</span>
            </span>
          }
        >
          <ButtonIcon
            icon="download"
            iconOnly={true}
            isDisabled={true}
            onClick={() => {
              logger.warn('Not supported yet.');
            }}
          />
        </ToolTip>
      </li>
    </SubTopBottomBarButtons>
  </SubTopBar>
);

SearchSubTopBar.defaultProps = {
  isMinMaxValuesByTarget: false,
  viewportChanged: false
};

SearchSubTopBar.propTypes = {
  isMinMaxValuesByTarget: PropTypes.bool,
  resetViewport: PropTypes.func.isRequired,
  normalize: PropTypes.func.isRequired,
  pubSub: PropTypes.object.isRequired,
  setShowAutoencodings: PropTypes.func,
  showAutoencodings: PropTypes.bool,
  viewportChanged: PropTypes.bool
};

const mapStateToProps = state => ({
  showAutoencodings: state.present.showAutoencodings
});
const mapDispatchToProps = dispatch => ({
  setShowAutoencodings: showAutoencodings =>
    dispatch(setShowAutoencodings(showAutoencodings))
});

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(withPubSub(SearchSubTopBar));
