import PropTypes from 'prop-types';
import React from 'react';
import { connect } from 'react-redux';

// Components
import ButtonIcon from './ButtonIcon';
import HiGlassViewer from './HiGlassViewer';
import ButtonRadio from './ButtonRadio';

// Configs
import {
  BUTTON_RADIO_CLASSIFICATION_OPTIONS,
  BLUE_PINK_CMAP,
  BLUE_PINK_TEXT_CMAP
} from '../configs/search';

import './HiglassResult.scss';

const getColor = prob =>
  BLUE_PINK_CMAP[Math.round(prob * BLUE_PINK_CMAP.length)];
const getFontColor = prob =>
  BLUE_PINK_TEXT_CMAP[Math.round(prob * BLUE_PINK_CMAP.length)];

class HiglassResult extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      isInfoSideBarShown: false,
      isMinMaxValuesByTarget: false
    };

    this.minMaxValues = {};

    this.initApi = false;
    this.onApiBnd = this.onApi.bind(this);
    this.onNormalizeBnd = this.onNormalize.bind(this);
    this.onToggleInfoSideBarBnd = this.onToggleInfoSideBar.bind(this);
    this.onEnterBnd = this.onEnter.bind(this);
    this.onLeaveBnd = this.onLeave.bind(this);
  }

  componentDidMount() {
    if (this.isToBeNormalized()) {
      this.normalize();
    }
  }

  componentDidUpdate(prevProps) {
    if (this.props.normalizeBy !== prevProps.normalizeBy) {
      this.normalize();
    }
  }

  /* ---------------------------- Getter & Setter --------------------------- */

  get viewId() {
    return `${this.props.searchId}.${this.props.windowId}.${
      this.props.showAutoencodings ? 'e' : ''
    }`;
  }

  /* ---------------------------- Custom Methods ---------------------------- */

  checkInitNormalize() {
    if (!this.initApi && this.isToBeNormalized()) {
      this.normalize();
    }
  }

  isToBeNormalized() {
    return Object.keys(this.props.normalizeBy)
      .map(key => this.props.normalizeBy[key])
      .reduce((a, b) => [...a, ...b], [])
      .some(x => x);
  }

  async normalize() {
    if (!this.api) return;

    if (this.props.normalizationSource !== this.props.windowId) {
      await this.setState({ isMinMaxValuesByTarget: false });
    }

    Object.keys(this.props.normalizeBy).forEach(track => {
      this.api.setTrackValueScaleLimits(
        undefined,
        track,
        ...this.props.normalizeBy[track]
      );
    });
  }

  onApi(api) {
    this.api = api;
    this.checkInitNormalize();
    this.initApi = true;
  }

  async onNormalize() {
    if (!this.api) return;

    this.minMaxValues = {};
    this.props.dataTracks.forEach(track => {
      if (this.state.isMinMaxValuesByTarget) {
        this.minMaxValues[track] = [undefined, undefined];
      } else {
        this.minMaxValues[track] = [
          0,
          this.api.getMinMaxValue(undefined, track, true)[1]
        ];
      }
    });

    await this.setState({
      isMinMaxValuesByTarget: !this.state.isMinMaxValuesByTarget
    });

    this.props.onNormalize(this.minMaxValues, this.props.windowId);
  }

  onToggleInfoSideBar() {
    this.setState({ isInfoSideBarShown: !this.state.isInfoSideBarShown });
  }

  onEnter() {
    if (this.props.onEnter) this.props.onEnter(this.props.windowId);
  }

  onLeave() {
    if (this.props.onLeave) this.props.onLeave();
  }

  /* -------------------------------- Render -------------------------------- */

  render() {
    let classNameInfoSideBar = 'higlass-result-side-panel';

    if (this.props.isInfoSideBar) {
      classNameInfoSideBar += ' higlass-result-has-info';
    }
    if (this.state.isInfoSideBarShown) {
      classNameInfoSideBar += ' higlass-result-show-info';
    }

    return (
      <div
        className="rel flex-c higlass-result"
        onMouseEnter={this.onEnterBnd}
        onMouseLeave={this.onLeaveBnd}
      >
        <aside className={classNameInfoSideBar}>
          {this.props.isInfoSideBar && (
            <ButtonIcon
              className="higlass-result-info-panel-toggler"
              icon="info"
              iconOnly={true}
              onClick={this.onToggleInfoSideBarBnd}
            />
          )}
          <ButtonIcon
            className="higlass-result-normalizer"
            icon="ratio"
            iconOnly={true}
            isActive={this.state.isMinMaxValuesByTarget}
            isIconMirrorOnFocus={true}
            onClick={this.onNormalizeBnd}
          />
          {this.props.isInfoSideBar && (
            <div className="full-dim higlass-result-info-panel-content">
              <ul className="no-list-style">
                {this.props.classificationProb && (
                  <li>
                    <label className="label">
                      Classification <abbr title="Probability">prob</abbr>
                    </label>
                    <div className="value">{this.props.classificationProb}</div>
                  </li>
                )}
              </ul>
            </div>
          )}
        </aside>
        <HiGlassViewer
          api={this.onApiBnd}
          height={this.props.viewHeight}
          isStatic={true}
          isZoomFixed={true}
          viewConfigId={this.viewId}
        />
        <div className="higlass-class-probability-wrapper">
          {!!this.props.classificationProb && (
            <div className="higlass-class-probability">
              <div
                className="higlass-class-probability-bar"
                style={{
                  bottom: `${this.props.classificationProb * 100}%`,
                  backgroundColor: getColor(this.props.classificationProb)
                }}
              />
              <div
                className="flex-c higlass-class-probability-label"
                style={{
                  bottom: `${this.props.classificationProb * 100}%`
                }}
              >
                <div
                  className="higlass-class-probability-label-prob"
                  style={{
                    color: getFontColor(this.props.classificationProb),
                    backgroundColor: getColor(this.props.classificationProb)
                  }}
                >
                  {this.props.classificationProb}
                </div>
                <div
                  className="higlass-class-probability-label-arrow"
                  style={{
                    borderLeftColor: getColor(this.props.classificationProb)
                  }}
                />
              </div>
            </div>
          )}
          <ButtonRadio
            isVertical={true}
            name={`search-${this.props.windowId}-classify`}
            onClick={this.props.classificationChangeHandler(
              this.props.windowId
            )}
            options={BUTTON_RADIO_CLASSIFICATION_OPTIONS}
            selection={
              (this.props.windows[this.props.windowId] &&
                this.props.windows[this.props.windowId].classification) ||
              this.props.classification
            }
            defaultSelection="neutral"
          />
        </div>
      </div>
    );
  }
}

HiglassResult.defaultProps = {
  classification: 'neutral',
  classificationProb: null,
  dataTracks: [],
  isInfoSideBar: false,
  normalizeBy: {},
  windows: {}
};

HiglassResult.propTypes = {
  classification: PropTypes.string,
  classificationProb: PropTypes.number,
  classificationChangeHandler: PropTypes.func.isRequired,
  dataTracks: PropTypes.array,
  isInfoSideBar: PropTypes.bool,
  normalizationSource: PropTypes.string,
  normalizeBy: PropTypes.object,
  onEnter: PropTypes.func,
  onLeave: PropTypes.func,
  onNormalize: PropTypes.func.isRequired,
  searchId: PropTypes.number.isRequired,
  showAutoencodings: PropTypes.bool.isRequired,
  viewHeight: PropTypes.number.isRequired,
  windowId: PropTypes.number.isRequired,
  windows: PropTypes.object
};

const mapStateToProps = state => ({
  showAutoencodings: state.present.showAutoencodings
});
const mapDispatchToProps = (/* dispatch */) => ({});

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(HiglassResult);
