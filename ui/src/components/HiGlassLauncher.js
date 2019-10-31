import { isEqual } from 'lodash-es';
import { HiGlassComponent } from 'higlass';
import PropTypes from 'prop-types';
import React from 'react';

// Utils
import {
  debounce,
  deepClone,
  Logger,
  removeHiGlassEventListeners
} from '../utils';

// Configs
import { SELECT } from '../configs/mouse-tools';
import { DEFAULT_OPTIONS } from '../configs/higlass';

// Styles
import './HiGlassLauncher.scss';

const logger = Logger('HiGlassLauncher'); // eslint-disable-line

class HiGlassLauncher extends React.Component {
  constructor(props) {
    super(props);

    this.hiGlassEventListeners = [];

    this.updateViewConfigDb = debounce(this.updateViewConfig.bind(this), 1000);
  }

  /* --------------------- React's Life Cycle Methods ----------------------- */

  componentWillUnmount() {
    removeHiGlassEventListeners(this.hiGlassEventListeners, this.api);
    this.hiGlassEventListeners = [];
  }

  shouldComponentUpdate(nextProps) {
    if (nextProps.mouseTool !== this.props.mouseTool) {
      this.setMouseTool(nextProps.mouseTool);
    }

    if (isEqual(this.newViewConfig, nextProps.viewConfig)) {
      return false;
    }

    return true;
  }

  componentDidMount() {
    if (this.hgc) this.registerHiGlassApi(this.hgc.api);
  }

  componentDidUpdate() {
    if (this.hgc) this.registerHiGlassApi(this.hgc.api);
  }

  /* ---------------------------- Custom Methods ---------------------------- */

  addHiGlassEventListeners() {
    if (!this.props.setViewConfig) return;

    this.hiGlassEventListeners.push({
      event: 'viewConfig',
      id: this.api.on('viewConfig', this.updateViewConfigDb)
    });
  }

  registerHiGlassApi(newApi) {
    if (this.api && this.api === newApi) return;
    if (this.api) {
      removeHiGlassEventListeners(this.hiGlassEventListeners, this.api);
    }
    this.api = newApi;
    this.addHiGlassEventListeners();
    this.props.api(this.api);
  }

  setMouseTool(mouseTool) {
    if (!this.props.enableAltMouseTools) return;

    switch (mouseTool) {
      case SELECT:
        this.api.activateTool('select');
        break;

      default:
        this.api.activateTool('move');
    }
  }

  updateViewConfig(newViewConfig) {
    this.newViewConfig = JSON.parse(newViewConfig);

    if (!isEqual(this.newViewConfig, this.props.viewConfig)) {
      this.props.setViewConfig(this.newViewConfig);
    }
  }

  /* -------------------------------- Render -------------------------------- */

  render() {
    const options = { ...DEFAULT_OPTIONS, ...this.props.options };

    if (this.props.enableAltMouseTools) {
      options.mouseTool = this.props.mouseTool;
    }

    options.globalMousePosition = this.props.isGlobalMousePosition;

    options.bounded = this.props.autoExpand
      ? false
      : this.props.options.bounded;

    if (this.props.isNotEditable) options.editable = false;

    if (this.props.isPixelPrecise) {
      options.bounded = false;
      options.pixelPreciseMarginPadding = true;
    }

    if (this.props.useCanvas) {
      options.renderer = 'canvas';
    }

    const className = !this.props.autoExpand ? 'full-dim' : 'rel';

    let classNameHgLauncher = 'higlass-launcher twbs';
    classNameHgLauncher += !this.props.autoExpand
      ? ' higlass-launcher-full'
      : '';
    classNameHgLauncher += this.props.isPadded
      ? ' higlass-launcher-padded'
      : '';

    return (
      <div className={className}>
        <div className={classNameHgLauncher}>
          <HiGlassComponent
            ref={c => {
              this.hgc = c;
            }}
            options={options || {}}
            viewConfig={deepClone(this.props.viewConfig)}
            zoomFixed={this.props.isZoomFixed}
          />
        </div>
      </div>
    );
  }
}

HiGlassLauncher.defaultProps = {
  disableTrackMenu: false,
  isGlobalMousePosition: false,
  isPixelPrecise: false,
  isZoomFixed: false,
  options: {
    bounded: true,
    horizontalMargin: 0,
    verticalMargin: 0
  },
  useCanvas: false
};

HiGlassLauncher.propTypes = {
  api: PropTypes.func,
  autoExpand: PropTypes.bool,
  enableAltMouseTools: PropTypes.bool,
  isGlobalMousePosition: PropTypes.bool,
  isNotEditable: PropTypes.bool,
  isPadded: PropTypes.bool,
  isPixelPrecise: PropTypes.bool,
  isZoomFixed: PropTypes.bool,
  mouseTool: PropTypes.string,
  onError: PropTypes.func.isRequired,
  options: PropTypes.object,
  setViewConfig: PropTypes.func,
  useCanvas: PropTypes.bool,
  viewConfig: PropTypes.object
};

export default HiGlassLauncher;
