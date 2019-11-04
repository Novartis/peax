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

  componentDidUpdate(prevProps, prevState) {
    if (this.props.viewConfigId !== prevProps.viewConfigId) {
      this.loadViewConfig();
    }
    if (this.props.viewConfigAdjustments !== prevProps.viewConfigAdjustments) {
      this.adjustViewConfig();
    }
    if (
      this.state.isLoading === false &&
      prevState.isLoading === true &&
      this.props.onLoaded
    ) {
      this.props.onLoaded();
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
    if (
      !viewConf ||
      !this.props.isStatic ||
      !this.props.viewConfigAdjustments.length
    )
      return viewConf;

    if (
      !this.props.viewConfigAdjustments.every(adjustment =>
        objectHas(viewConf, adjustment.key)
      )
    ) {
      return viewConf;
    }

    const newViewConf = { ...viewConf };

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
                containerPadding={this.props.containerPadding}
                enableAltMouseTools={this.props.enableAltMouseTools}
                onError={this.onError.bind(this)}
                viewConfig={this.state.viewConfigStatic}
                isGlobalMousePosition={this.props.isGlobalMousePosition}
                isPadded={this.props.isPadded}
                isNotEditable={this.props.isNotEditable}
                isZoomFixed={this.props.isZoomFixed}
                isPixelPrecise={this.props.isPixelPrecise}
                sizeMode={this.props.sizeMode}
                useCanvas={this.props.useCanvas}
                viewMargin={this.props.viewMargin}
                viewPadding={this.props.viewPadding}
              />
            ) : (
              <HiGlassLoader
                api={this.props.api}
                containerPadding={this.props.containerPadding}
                enableAltMouseTools={this.props.enableAltMouseTools}
                onError={this.onError.bind(this)}
                isGlobalMousePosition={this.props.isGlobalMousePosition}
                isPadded={this.props.isPadded}
                isNotEditable={this.props.isNotEditable}
                isZoomFixed={this.props.isZoomFixed}
                isPixelPrecise={this.props.isPixelPrecise}
                sizeMode={this.props.sizeMode}
                useCanvas={this.props.useCanvas}
                viewMargin={this.props.viewMargin}
                viewPadding={this.props.viewPadding}
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
  isPixelPrecise: false,
  useCanvas: false,
  viewConfigAdjustments: []
};

HiGlassViewer.propTypes = {
  api: PropTypes.func,
  autoExpand: PropTypes.bool,
  className: PropTypes.string,
  containerPadding: PropTypes.oneOfType([PropTypes.number, PropTypes.array]),
  enableAltMouseTools: PropTypes.bool,
  hasSubTopBar: PropTypes.bool,
  height: PropTypes.number,
  isGlobalMousePosition: PropTypes.bool,
  isNotEditable: PropTypes.bool,
  isPadded: PropTypes.bool,
  isStatic: PropTypes.bool,
  isZoomFixed: PropTypes.bool,
  isPixelPrecise: PropTypes.bool,
  isScrollable: PropTypes.bool,
  onLoaded: PropTypes.func,
  pubSub: PropTypes.object.isRequired,
  setViewConfig: PropTypes.func.isRequired,
  sizeMode: PropTypes.string,
  useCanvas: PropTypes.bool,
  viewConfig: PropTypes.object,
  viewConfigId: PropTypes.string,
  viewConfigAdjustments: PropTypes.array,
  viewMargin: PropTypes.oneOfType([PropTypes.number, PropTypes.array]),
  viewPadding: PropTypes.oneOfType([PropTypes.number, PropTypes.array])
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
