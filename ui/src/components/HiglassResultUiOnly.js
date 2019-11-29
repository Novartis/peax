import { boundMethod } from 'autobind-decorator';
import PropTypes from 'prop-types';
import React from 'react';
import { connect } from 'react-redux';

// Components
import ButtonIcon from './ButtonIcon';
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
  BLUE_PINK_CMAP[Math.round(prob * (BLUE_PINK_CMAP.length - 1))];
const getFontColor = prob =>
  BLUE_PINK_TEXT_CMAP[Math.round(prob * (BLUE_PINK_CMAP.length - 1))];

class HiglassResultUiOnly extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      isInfoSideBarShown: false,
      isMinMaxValuesByTarget: false
    };

    this.minMaxValues = {};

    this.initApi = false;
    this.checkApi();
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
    if (this.props.hgApi !== prevProps.hgApi) {
      this.checkApi();
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
    if (!this.props.hgApi) return;

    if (this.props.normalizationSource !== this.props.windowId) {
      await this.setState({ isMinMaxValuesByTarget: false });
    }

    Object.keys(this.props.normalizeBy).forEach(track => {
      this.props.hgApi.setTrackValueScaleLimits(
        `view${this.props.viewNumber + 1}`,
        track,
        ...this.props.normalizeBy[track]
      );
    });
  }

  checkApi() {
    if (!this.props.hgApi) return;
    this.checkInitNormalize();
    this.initApi = true;
  }

  @boundMethod
  async onNormalize() {
    if (!this.props.hgApi) return;

    this.minMaxValues = {};

    let absMax = -Infinity;
    this.props.dataTracks.forEach(track => {
      if (this.state.isMinMaxValuesByTarget) {
        this.minMaxValues[track] = [undefined, undefined];
      } else {
        const maxValue = this.props.hgApi.getMinMaxValue(
          `view${this.props.viewNumber + 1}`,
          track,
          true
        )[1];
        absMax = Math.max(absMax, maxValue);
        this.minMaxValues[track] = [0, maxValue];
      }
    });

    // When value scales are locked already we need to use the group wise min-max
    // values rather than track specific values.
    if (this.props.valueScalesLocked && !this.state.isMinMaxValuesByTarget) {
      this.props.dataTracks.forEach(track => {
        this.minMaxValues[track] = [0, absMax];
      });
    }

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
    let className = 'rel flex-c higlass-result single-higlass-instance';

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
            <strong>Potential false negative: </strong>
            Window labeled positive but the prediction probability is only{' '}
            <span className="prob">
              {Number.parseFloat(this.props.classificationProb).toFixed(2)}
            </span>
            !
          </div>
        )}
        {this.props.conflict === 'fp' && (
          <div className="conflict conflict-fp">
            <strong>Potential false positive: </strong>
            Window labeled negative but the prediction probability is{' '}
            <span className="prob">
              {Number.parseFloat(this.props.classificationProb).toFixed(2)}
            </span>
            !
          </div>
        )}
        <div
          className={className}
          onMouseEnter={this.onEnter}
          onMouseLeave={this.onLeave}
          style={{ height: `${this.props.viewHeight}px` }}
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
          <div
            className="higlass-mouse-move-container"
            onMouseMove={this.props.onMouseMove}
            onMouseOut={this.props.onMouseOut}
            onMouseOver={this.props.onMouseOver}
          />
          <div className="higlass-class-probability-wrapper">
            {!this.props.hidePredProb &&
              (this.props.classificationProb ? (
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
                      {Number(this.props.classificationProb).toFixed(
                        this.props.classificationProb < 1 ? 2 : 1
                      )}
                    </div>
                    <div
                      className="higlass-class-probability-label-arrow"
                      style={{
                        borderLeftColor: getColor(this.props.classificationProb)
                      }}
                    />
                  </div>
                </div>
              ) : (
                <div className="higlass-class-probability higlass-class-probability-unkonw">
                  <div className="flex-c higlass-class-probability-label">
                    <div className="higlass-class-probability-label-prob">
                      Prediction probability is below the threshold
                    </div>
                  </div>
                </div>
              ))}
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

HiglassResultUiOnly.defaultProps = {
  classification: 'neutral',
  classificationProb: null,
  dataTracks: [],
  hidePredProb: false,
  isInfoSideBar: false,
  normalizeBy: {},
  onEnter: toVoid,
  onLeave: toVoid,
  windows: {}
};

HiglassResultUiOnly.propTypes = {
  classification: PropTypes.string,
  classificationProb: PropTypes.number,
  classificationChangeHandler: PropTypes.func.isRequired,
  conflict: PropTypes.string,
  dataTracks: PropTypes.array,
  hgApi: PropTypes.object,
  hover: PropTypes.number.isRequired,
  hidePredProb: PropTypes.bool,
  isInfoSideBar: PropTypes.bool,
  normalizationSource: PropTypes.oneOfType([
    PropTypes.string,
    PropTypes.number
  ]),
  normalizeBy: PropTypes.object,
  onMouseMove: PropTypes.func.isRequired,
  onMouseOut: PropTypes.func.isRequired,
  onMouseOver: PropTypes.func.isRequired,
  onNormalize: PropTypes.func.isRequired,
  searchId: PropTypes.number.isRequired,
  selection: PropTypes.array.isRequired,
  setHover: PropTypes.func.isRequired,
  setSelection: PropTypes.func.isRequired,
  showAutoencodings: PropTypes.bool.isRequired,
  valueScalesLocked: PropTypes.bool,
  viewHeight: PropTypes.number.isRequired,
  viewNumber: PropTypes.number.isRequired,
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

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(HiglassResultUiOnly);
