import { boundMethod } from 'autobind-decorator';
import PropTypes from 'prop-types';
import React from 'react';
import { connect } from 'react-redux';

// Components
import ButtonIcon from './ButtonIcon';
import HiGlassViewer from './HiGlassViewer';
import ButtonRadio from './ButtonRadio';

// Utils
import { toVoid } from '../utils';

// Configs
import {
  BUTTON_RADIO_CLASSIFICATION_OPTIONS,
  BLUE_PINK_CMAP,
  BLUE_PINK_TEXT_CMAP
} from '../configs/search';

// Actions
import { setSearchHover, setSearchSelection } from '../actions';

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

  get isHovered() {
    return this.props.hover === this.props.windowId;
  }

  get isSelected() {
    return this.props.selection.indexOf(this.props.windowId) >= 0;
  }

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

  @boundMethod
  onApi(api) {
    this.api = api;
    this.checkInitNormalize();
    this.initApi = true;
  }

  @boundMethod
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

  @boundMethod
  onToggleInfoSideBar() {
    this.setState({ isInfoSideBarShown: !this.state.isInfoSideBarShown });
  }

  @boundMethod
  onEnter() {
    this.props.setHover(this.props.windowId);
  }

  @boundMethod
  onLeave() {
    this.props.setHover(-1);
  }

  @boundMethod
  onSelect() {
    const windowIndex = this.props.selection.indexOf(this.props.windowId);
    const newSelection = [...this.props.selection];
    if (windowIndex >= 0) {
      newSelection.splice(windowIndex, 1);
      this.props.setSelection(newSelection);
    } else {
      newSelection.push(this.props.windowId);
      this.props.setSelection(newSelection);
    }
  }

  /* -------------------------------- Render -------------------------------- */

  render() {
    let className = 'rel flex-c higlass-result';

    // if (this.isHovered) className += ' is-hovered';
    if (this.isSelected) className += ' is-selected';

    let classNameInfoSideBar = 'higlass-result-side-panel';

    if (this.props.isInfoSideBar) {
      classNameInfoSideBar += ' higlass-result-has-info';
    }
    if (this.state.isInfoSideBarShown) {
      classNameInfoSideBar += ' higlass-result-show-info';
    }

    return (
      <div>
        {this.props.conflict === 'fn' && (
          <div className="conflict conflict-fn">
            <strong>Potential false negative</strong>
            <p>
              Window labeled positive but the prediction probability is only{' '}
              <span className="prob">{this.props.classificationProb}</span>!
            </p>
          </div>
        )}
        {this.props.conflict === 'fp' && (
          <div className="conflict conflict-fp">
            <strong>Potential false positive</strong>
            <p>
              Window labeled negative but the prediction probability is{' '}
              <span className="prob">{this.props.classificationProb}</span>!
            </p>
          </div>
        )}
        <div
          className={className}
          onMouseEnter={this.onEnter}
          onMouseLeave={this.onLeave}
        >
          <aside className={classNameInfoSideBar}>
            {this.props.isInfoSideBar && (
              <ButtonIcon
                className="higlass-result-info-panel-toggler"
                icon="info"
                iconOnly={true}
                onClick={this.onToggleInfoSideBar}
              />
            )}
            <ButtonIcon
              className="higlass-result-selector"
              icon={this.isSelected ? 'circle-nested' : 'circle-hollow'}
              iconOnly={true}
              isActive={this.isSelected}
              onClick={this.onSelect}
            />
            <ButtonIcon
              className="higlass-result-normalizer"
              icon="ratio"
              iconOnly={true}
              isActive={this.state.isMinMaxValuesByTarget}
              isIconMirrorOnFocus={true}
              onClick={this.onNormalize}
            />
            {this.props.isInfoSideBar && (
              <div className="full-dim higlass-result-info-panel-content">
                <ul className="no-list-style">
                  {this.props.classificationProb && (
                    <li>
                      <label className="label">
                        Classification <abbr title="Probability">prob</abbr>
                      </label>
                      <div className="value">
                        {this.props.classificationProb}
                      </div>
                    </li>
                  )}
                </ul>
              </div>
            )}
          </aside>
          <HiGlassViewer
            api={this.onApi}
            height={this.props.viewHeight}
            isGlobalMousePosition
            isNotEditable
            isStatic
            isZoomFixed
            isPixelPrecise
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
  onEnter: toVoid,
  onLeave: toVoid,
  windows: {}
};

HiglassResult.propTypes = {
  classification: PropTypes.string,
  classificationProb: PropTypes.number,
  classificationChangeHandler: PropTypes.func.isRequired,
  conflict: PropTypes.string,
  dataTracks: PropTypes.array,
  hover: PropTypes.number.isRequired,
  isInfoSideBar: PropTypes.bool,
  normalizationSource: PropTypes.oneOfType([
    PropTypes.string,
    PropTypes.number
  ]),
  normalizeBy: PropTypes.object,
  onNormalize: PropTypes.func.isRequired,
  searchId: PropTypes.number.isRequired,
  selection: PropTypes.array.isRequired,
  setHover: PropTypes.func.isRequired,
  setSelection: PropTypes.func.isRequired,
  showAutoencodings: PropTypes.bool.isRequired,
  viewHeight: PropTypes.number.isRequired,
  windowId: PropTypes.number.isRequired,
  windows: PropTypes.object
};

const mapStateToProps = state => ({
  hover: state.present.searchHover,
  selection: state.present.searchSelection,
  showAutoencodings: state.present.showAutoencodings
});

const mapDispatchToProps = dispatch => ({
  setHover: windowId => dispatch(setSearchHover(windowId)),
  setSelection: windowIds => dispatch(setSearchSelection(windowIds))
});

export default connect(mapStateToProps, mapDispatchToProps)(HiglassResult);
