import PropTypes from 'prop-types';
import React from 'react';
import { connect } from 'react-redux';

// Higher-order components
import { withPubSub } from '../hocs/pub-sub';

// Components
import MessageCenter from './MessageCenter';
import HiGlassLauncher from './HiGlassLauncher';
import SpinnerCenter from './SpinnerCenter';

// Containers
import HiGlassLoader from '../containers/HiGlassLoader';

// Actions
import { setViewConfig } from '../actions';

// Utils
import { Deferred, getServer, Logger, objectHas, objectSet } from '../utils';

// Styles
import './HiGlassViewer.scss';

const logger = Logger('HiGlassViewer');

const fetchViewConfig = (configId, base = getServer()) =>
  fetch(`${base}/api/v1/viewconfs/?d=${configId}`)
    .then(response => response.text())
    .then(viewConfig =>
      JSON.parse(viewConfig.replace(/\/\/localhost:5000/g, getServer()))
    );

const defaultViewConfigId = 'default';

class HiGlassViewer extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      error: '',
      isLoading: true
    };
  }

  componentDidMount() {
    this.loadViewConfig();
  }

  componentDidUpdate(prevProps) {
    if (this.props.viewConfigId !== prevProps.viewConfigId) {
      this.loadViewConfig();
    }
    if (this.props.viewConfigAdjustments !== prevProps.viewConfigAdjustments) {
      this.adjustViewConfig();
    }
  }

  /* ---------------------------- Custom Methods ---------------------------- */

  confirmViewConfigChange() {
    const dialog = new Deferred();
    this.props.pubSub.publish('globalDialog', {
      message: 'You are about to override the existing view config.',
      request: dialog,
      rejectText: 'Cancel',
      resolveText: 'Okay'
    });
  }

  loadViewConfig(viewConfigId = this.props.viewConfigId) {
    // Make sure we always load the default view config
    if (!viewConfigId && this.props.viewConfig) {
      this.setState({
        error: '',
        isLoading: false
      });
      return;
    }

    this.setState({
      error: '',
      isLoading: true
    });

    fetchViewConfig(viewConfigId || defaultViewConfigId)
      .catch(() => {
        logger.warn('View config is not available locally!');

        // Try loading config from HiGlass.io
        return fetchViewConfig(
          viewConfigId || defaultViewConfigId,
          '//higlass.io'
        );
      })
      .then(this.setViewConfig.bind(this))
      .catch(error => {
        logger.error('Could not load or parse config.', error);
        this.setState({
          error: 'Could not load config.',
          isLoading: false
        });
      });
  }

  adjustViewConfig(
    viewConf = this.state.viewConfigStatic,
    doNotSetState = false
  ) {
    if (!this.props.isStatic || !this.props.viewConfigAdjustments.length)
      return viewConf;

    if (
      !this.props.viewConfigAdjustments.every(adjustment =>
        objectHas(viewConf, adjustment.key)
      )
    ) {
      return viewConf;
    }

    const newViewConf = Object.assign({}, viewConf);

    this.props.viewConfigAdjustments.forEach(adjustment =>
      objectSet(newViewConf, adjustment.key, adjustment.value)
    );

    if (!doNotSetState) {
      this.setState({
        viewConfigStatic: newViewConf
      });
    }

    return newViewConf;
  }

  onError(error) {
    this.setState({
      error,
      isLoading: false
    });
  }

  setViewConfig(viewConfig) {
    if (!viewConfig || viewConfig.error) {
      const errorMsg =
        viewConfig && viewConfig.error
          ? viewConfig.error
          : 'View config broken.';
      this.setState({
        error: errorMsg,
        isLoading: false
      });
    } else if (this.props.isStatic) {
      this.setState({
        error: '',
        isLoading: false,
        viewConfigStatic: this.adjustViewConfig(viewConfig, true)
      });
    } else {
      this.props.setViewConfig(viewConfig);
      this.setState({
        error: '',
        isLoading: false
      });
    }
  }

  /* -------------------------------- Render -------------------------------- */

  render() {
    let className = 'higlass-viewer';

    className += this.props.hasSubTopBar ? ' has-sub-top-bar' : '';
    className += this.props.height ? ' higlass-viewer-abs-height' : ' full-dim';
    className += ` ${this.props.className || ''}`;

    const style = {
      height: this.props.height ? `${this.props.height}px` : 'auto'
    };

    return (
      <div className={className} style={style}>
        <div className="higlass-viewer-padded-container">
          {this.state.error && (
            <MessageCenter msg={this.state.error} type="error" />
          )}
          {!this.state.error &&
            (this.state.isLoading ? ( // eslint-disable-line no-nested-ternary
              <SpinnerCenter />
            ) : this.props.isStatic ? (
              <HiGlassLauncher
                api={this.props.api}
                autoExpand={this.props.autoExpand}
                disableTrackMenu={this.props.disableTrackMenu}
                enableAltMouseTools={this.props.enableAltMouseTools}
                onError={this.onError.bind(this)}
                viewConfig={this.state.viewConfigStatic}
                isGlobalMousePosition={this.props.isGlobalMousePosition}
                isPadded={this.props.isPadded}
                isZoomFixed={this.props.isZoomFixed}
              />
            ) : (
              <HiGlassLoader
                api={this.props.api}
                disableTrackMenu={this.props.disableTrackMenu}
                enableAltMouseTools={this.props.enableAltMouseTools}
                onError={this.onError.bind(this)}
                isGlobalMousePosition={this.props.isGlobalMousePosition}
                isPadded={this.props.isPadded}
                isZoomFixed={this.props.isZoomFixed}
              />
            ))}
        </div>
      </div>
    );
  }
}

HiGlassViewer.defaultProps = {
  api: () => {},
  disableTrackMenu: false,
  isGlobalMousePosition: false,
  isPadded: false,
  isStatic: false,
  isZoomFixed: false,
  viewConfigAdjustments: []
};

HiGlassViewer.propTypes = {
  api: PropTypes.func,
  autoExpand: PropTypes.bool,
  className: PropTypes.string,
  disableTrackMenu: PropTypes.bool,
  enableAltMouseTools: PropTypes.bool,
  hasSubTopBar: PropTypes.bool,
  height: PropTypes.number,
  isGlobalMousePosition: PropTypes.bool,
  isPadded: PropTypes.bool,
  isStatic: PropTypes.bool,
  isZoomFixed: PropTypes.bool,
  pubSub: PropTypes.object.isRequired,
  setViewConfig: PropTypes.func.isRequired,
  viewConfig: PropTypes.object,
  viewConfigId: PropTypes.string,
  viewConfigAdjustments: PropTypes.array
};

const mapStateToProps = state => ({
  viewConfig: state.present.viewConfig
});

const mapDispatchToProps = dispatch => ({
  setViewConfig: viewConfig => {
    dispatch(setViewConfig(viewConfig));
  }
});

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(withPubSub(HiGlassViewer));
