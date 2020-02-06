import { boundMethod } from 'autobind-decorator';
import PropTypes from 'prop-types';
import React from 'react';
import { connect } from 'react-redux';
import { compose } from 'recompose';
import createScatterplot from 'regl-scatterplot';

// Components
import Badge from '../components/Badge';
import BarChart from '../components/BarChart';
import Button from '../components/Button';
import ButtonIcon from '../components/ButtonIcon';
import ButtonRadio from '../components/ButtonRadio';
import ElementWrapperAdvanced from '../components/ElementWrapperAdvanced';
import SearchUncertaintyHelp from '../components/SearchUncertaintyHelp';
import SearchPredPropChangeHelp from '../components/SearchPredPropChangeHelp';
import SearchDivergenceHelp from '../components/SearchDivergenceHelp';
import LabeledSlider from '../components/LabeledSlider';
import TabEntry from '../components/TabEntry';

// Actions
import {
  setSearchHover,
  setSearchRightBarMetadata,
  setSearchRightBarProgress,
  setSearchRightBarProjection,
  setSearchRightBarProjectionSettings,
  setSearchSelection
} from '../actions';

// Utils
import {
  api,
  debounce,
  Deferred,
  inputToNum,
  readableDate,
  zip
} from '../utils';

// Configs
import {
  COLOR_BG,
  COLORMAP_CAT,
  COLORMAP_PRB,
  HOVER_DELAY,
  REDRAW_DELAY,
  PROJECTION_CHECK_INTERVAL,
  PROJECTION_VIEW,
  PROJECTION_VIEW_INTERVAL,
  SHOW_RECTICLE,
  RECTICLE_COLOR
} from '../configs/projection';

import {
  BUTTON_RADIO_PROJECTION_COLOR_ENCODING_OPTIONS,
  TAB_RIGHT_BAR_INFO
} from '../configs/search';

// Styles
import './Search.scss';

const showUncertaintyHelp = pubSub => () => {
  pubSub.publish('globalDialog', {
    message: <SearchUncertaintyHelp />,
    request: new Deferred(),
    resolveOnly: true,
    resolveText: 'Close',
    icon: 'help-circle',
    headline: 'Uncertainty Plot'
  });
};

const showPredPropChangeHelp = pubSub => () => {
  pubSub.publish('globalDialog', {
    message: <SearchPredPropChangeHelp />,
    request: new Deferred(),
    resolveOnly: true,
    resolveText: 'Close',
    icon: 'help-circle',
    headline: 'Prediction Probability Change Plot'
  });
};

const showPredProbDivergenceHelp = pubSub => () => {
  pubSub.publish('globalDialog', {
    message: <SearchDivergenceHelp />,
    request: new Deferred(),
    resolveOnly: true,
    resolveText: 'Close',
    icon: 'help-circle',
    headline: 'Convergence/Divergence Plot'
  });
};

class SearchRightBarInfo extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      canvas: null,
      colorEncoding: 'categorical',
      hoveredPoint: -1,
      isColorByProb: false,
      isDefaultView: true,
      isError: false,
      isInit: false,
      isLoading: false,
      pointSize: 3,
      points: [],
      scatterplot: null,
      selected: [],
      settingsUmapMinDist: 0.1,
      settingsUmapNN: 5,
      umapMinDist: 0.1,
      umapNN: 5
    };

    this.onChangeColorEncoding = compose(this.onChangeState('colorEncoding'));
    this.onChangePointSize = compose(
      this.onChangeState('pointSize'),
      pointSize => {
        if (this.scatterplot) {
          this.scatterplot.set({ pointSize });
        }
        return pointSize;
      },
      inputToNum
    );
    this.onChangeSettingsUmapNN = compose(
      this.onChangeState('settingsUmapNN'),
      inputToNum
    );
    this.onChangeSettingsUmapMinDist = compose(
      this.onChangeState('settingsUmapMinDist'),
      inputToNum
    );

    this.checkViewDb = debounce(
      this.checkView.bind(this),
      PROJECTION_VIEW_INTERVAL
    );
    this.drawScatterplotDb = debounce(
      this.drawScatterplot.bind(this),
      REDRAW_DELAY
    );
    this.onPointOverDb = debounce(this.onPointOver, HOVER_DELAY);
    this.onPointOutDb = debounce(this.onPointOut, HOVER_DELAY);
  }

  componentDidMount() {
    if (!Number.isNaN(+this.props.searchInfo.id)) this.loadProjection();
  }

  componentDidUpdate(prevProps, prevState) {
    if (
      this.isOpen &&
      this.props.searchInfo.id &&
      (!this.state.isInit ||
        this.props.searchInfo.classifiers !== prevProps.searchInfo.classifiers)
    )
      this.loadProjection();
    if (
      this.isOpen &&
      (this.props.tab !== prevProps.tab ||
        this.state.points !== prevState.points ||
        (this.props.barWidth && this.props.barWidth !== prevProps.barWidth))
    )
      this.drawScatterplotDb(true);
    if (this.state.colorEncoding !== prevState.colorEncoding)
      this.setColorEncoding();
    if (this.props.selection !== this.selection) this.select();
    if (this.props.hover !== this.hoveredPoint) this.hover();
  }

  componentWillUnmount() {
    if (this.scatterplot) this.scatterplot.destroy();
  }

  /* ---------------------------- Custom Methods ---------------------------- */

  get isOpen() {
    return this.props.barShow && this.props.tab === TAB_RIGHT_BAR_INFO;
  }

  select() {
    if (!this.scatterplot) return;
    if (this.props.selection.length) {
      this.scatterplot.select(this.props.selection);
    } else {
      this.scatterplot.deselect();
    }
  }

  hover() {
    if (!this.scatterplot) return;
    if (this.props.hover >= 0) {
      this.scatterplot.hover(this.props.hover, true);
    } else {
      this.scatterplot.hover();
    }
  }

  setColorEncoding() {
    if (!this.scatterplot) return;
    if (this.state.colorEncoding === 'probability') {
      this.scatterplot.set({ colorBy: 'value', colors: COLORMAP_PRB });
    } else {
      this.scatterplot.set({ colorBy: 'category', colors: COLORMAP_CAT });
    }
  }

  drawScatterplot(updateSize = false) {
    if (!this.scatterplot) this.initScatterplot();
    else this.updateScatterplot(this.state.points, updateSize);
  }

  initScatterplot(points = this.state.points) {
    if (!points.length || !this.canvasWrapper) return;

    const bBox = this.canvasWrapper.getBoundingClientRect();

    const scatterplot = createScatterplot({
      background: COLOR_BG,
      width: bBox.width,
      height: bBox.height,
      pointSize: this.state.pointSize,
      showRecticle: SHOW_RECTICLE,
      recticleColor: RECTICLE_COLOR,
      view: PROJECTION_VIEW
    });

    scatterplot.subscribe('pointover', this.onPointOverDb);
    scatterplot.subscribe('pointout', this.onPointOutDb);
    scatterplot.subscribe('view', this.checkViewDb);
    scatterplot.subscribe('select', this.onSelect);
    scatterplot.subscribe('deselect', this.onDeselect);

    scatterplot.draw(points);

    this.scatterplot = scatterplot;
    this.setColorEncoding();

    this.setState({ canvas: scatterplot.get('canvas') });
  }

  updateScatterplot(points = this.state.points, updateSize = false) {
    if (!points.length || !this.canvasWrapper) return;
    if (updateSize) {
      const bBox = this.canvasWrapper.getBoundingClientRect();
      this.scatterplot.set({
        width: bBox.width,
        height: bBox.height
      });
      this.scatterplot.refresh();
    }
    this.scatterplot.draw(points);
  }

  prepareProjection(projection, classes, probabilities) {
    return zip([projection, classes, probabilities], [2, 1, 1]);
  }

  @boundMethod
  async newProjection() {
    if (this.state.isLoading) return;

    this.setState({ isLoading: true, isError: false, isNotFound: false });

    const resp = await api.newProjection(
      this.props.searchInfo.id,
      this.state.settingsUmapMinDist,
      this.state.settingsUmapNN
    );
    const isError = resp.status !== 200 ? "Couldn't project data." : false;

    await this.setState({
      umapMinDist: this.state.settingsUmapMinDist,
      umapNN: this.state.settingsUmapNN
    });

    const checkProjectionTimer = isError
      ? null
      : setInterval(this.checkProjection, PROJECTION_CHECK_INTERVAL);

    this.setState({
      isError,
      checkProjectionTimer
    });
  }

  @boundMethod
  async checkProjection() {
    const resp = await api.newProjection(
      this.props.searchInfo.id,
      this.state.umapMinDist,
      this.state.umapNN
    );
    const isError = resp.status !== 200 ? "Couldn't project data." : false;
    const projection = isError ? {} : resp.body;

    if (projection.projectorIsFitting || projection.projectionIsProjecting)
      return;

    clearInterval(this.state.checkProjectionTimer);

    if (isError) this.setState({ isError });
    else this.loadProjection(true);
  }

  async loadProjection(force = false) {
    if (this.state.isLoading && !force) return;

    this.setState({ isLoading: true, isError: false });

    const respClasses = await api.getClasses(this.props.searchInfo.id);
    const respProbs = await api.getProbabilities(this.props.searchInfo.id);
    const respProj = await api.getProjection(this.props.searchInfo.id);

    let isError =
      respClasses.status !== 200 || respProbs.status !== 200 || respProj !== 200
        ? 'Error'
        : false;

    // Compare the number of windows for which we got classifications, projections, and
    // projections. Those should be the same otherwise the data seems to be corrupted
    const numDiffLenghts = !isError
      ? new Set([
          respClasses.body.results.length,
          respProbs.body.results.length,
          respProj.body.projection.length / 2
        ]).size
      : new Set();

    const isNotFound =
      respProj.status === 404 ? 'Projection not computed.' : false;

    isError = !isNotFound && numDiffLenghts > 1 ? 'Data is corrupted!' : false;

    isError =
      !isError &&
      !isNotFound &&
      (respClasses.status !== 200 ||
        respProbs.status !== 200 ||
        respProj.status !== 200)
        ? "Couldn't load projection."
        : isError;

    const classes = isNotFound || isError ? [] : respClasses.body.results;
    const probabilities = isNotFound || isError ? [] : respProbs.body.results;
    const projection = isNotFound || isError ? [] : respProj.body.projection;
    const points = this.prepareProjection(projection, classes, probabilities);
    const umapMinDist =
      isNotFound || isError
        ? this.state.umapMinDist
        : respProj.body.projectorSettings.min_dist;
    const umapNN =
      isNotFound || isError
        ? this.state.umapNN
        : respProj.body.projectorSettings.n_neighbors;

    this.setState({
      isNotFound,
      isError,
      isInit: true,
      isLoading: false,
      points,
      umapMinDist,
      umapNN,
      settingsUmapMinDist: umapMinDist,
      settingsUmapNN: umapNN
    });
  }

  onChangeSettingsNN(key, value) {
    this.setState({ [key]: value });
  }

  onChangeState(key) {
    return value => {
      this.setState({ [key]: value });
    };
  }

  @boundMethod
  onRef(canvasWrapper) {
    this.canvasWrapper = canvasWrapper;
  }

  @boundMethod
  onPointOver(hoveredPoint) {
    // We need to store a reference to this object to avoid circular events
    this.hoveredPoint = hoveredPoint;
    this.props.setHover(hoveredPoint);
  }

  @boundMethod
  onPointOut() {
    this.hoveredPoint = -1;
    this.props.setHover(-1);
  }

  @boundMethod
  onSelect({ points: selectedPoints = [] } = {}) {
    // We need to store a reference to this object to avoid circular events
    this.selection = selectedPoints;
    this.props.setSelection(selectedPoints);
  }

  @boundMethod
  onDeselect() {
    this.props.setSelection([]);
  }

  checkView(view) {
    const isDefaultView = view.every(
      (x, i) => Math.round(x * 1000000) === PROJECTION_VIEW[i] * 1000000
    );
    this.setState({ isDefaultView });
  }

  @boundMethod
  onResetLocation() {
    this.scatterplot.reset();
  }

  /* -------------------------------- Render -------------------------------- */

  render() {
    const isHovered = this.hoveredPoint !== null && +this.hoveredPoint >= 0;

    const posActive =
      isHovered && this.state.points[this.hoveredPoint][2] === 2;
    const negActive =
      isHovered && this.state.points[this.hoveredPoint][2] === 1;
    const unlabeledActive =
      isHovered && this.state.points[this.hoveredPoint][2] === 0;
    const targetActive =
      isHovered && this.state.points[this.hoveredPoint][2] === 3;

    const predProb1Active =
      isHovered && this.state.points[this.hoveredPoint][3] <= 1 / 7;
    const predProb2Active =
      isHovered &&
      this.state.points[this.hoveredPoint][3] > 1 / 7 &&
      this.state.points[this.hoveredPoint][3] <= 2 / 7;
    const predProb3Active =
      isHovered &&
      this.state.points[this.hoveredPoint][3] > 2 / 7 &&
      this.state.points[this.hoveredPoint][3] <= 3 / 7;
    const predProb4Active =
      isHovered &&
      this.state.points[this.hoveredPoint][3] > 3 / 7 &&
      this.state.points[this.hoveredPoint][3] <= 4 / 7;
    const predProb5Active =
      isHovered &&
      this.state.points[this.hoveredPoint][3] > 4 / 7 &&
      this.state.points[this.hoveredPoint][3] <= 5 / 7;
    const predProb6Active =
      isHovered &&
      this.state.points[this.hoveredPoint][3] > 5 / 7 &&
      this.state.points[this.hoveredPoint][3] <= 6 / 7;
    const predProb7Active =
      isHovered && this.state.points[this.hoveredPoint][3] > 6 / 7;

    return (
      <div className="right-bar-info flex-c flex-v full-wh">
        <TabEntry
          isOpen={this.props.showProjection}
          title="Projection"
          toggle={this.props.toggleProjection}
        >
          <div className="search-right-bar-padding">
            <div
              className="search-projection-wrapper"
              ref={this.onRef}
              onMouseDown={event => {
                if (event.detail > 1) {
                  // Prevent text select of outside elements on double click
                  event.preventDefault();
                }
              }}
            >
              {this.isOpen && (
                <ElementWrapperAdvanced
                  className="search-projection"
                  element={this.state.canvas}
                  isError={this.state.isError}
                  isErrorNodes={
                    <Button onClick={this.newProjection}>{'Re-compute'}</Button>
                  }
                  isLoading={this.state.isLoading}
                  isNotFound={
                    this.state.isNotFound && (
                      <Button onClick={this.newProjection} isPrimary>
                        Compute Projection
                      </Button>
                    )
                  }
                  onAppend={() => {
                    this.scatterplot.refresh();
                  }}
                />
              )}
              {!this.state.isDefaultView && (
                <ButtonIcon
                  className="search-projection-reset"
                  icon="reset"
                  iconOnly={true}
                  isIconRotationOnFocus={true}
                  isDisabled={this.state.isDefaultView}
                  onClick={this.onResetLocation}
                />
              )}
            </div>
            {!!this.state.points.length && (
              <ul className="r no-list-style compact-list right-bar-v-padding">
                <li className="flex-c flex-jc-sb">
                  <ButtonRadio
                    className="full-w"
                    name="search-projection-color-encoding"
                    onClick={this.onChangeColorEncoding}
                    options={BUTTON_RADIO_PROJECTION_COLOR_ENCODING_OPTIONS}
                    selection={this.state.colorEncoding}
                  />
                </li>
                {this.state.colorEncoding === 'probability' ? (
                  <li className="m-t-0-25">
                    <ul className="no-list-style flex-c flex-jc-sb colormap">
                      <li className="flex-g-1 colormap-0-label" />
                      <li className="flex-g-1 colormap-3-label" />
                      <li className="flex-g-1 colormap-6-label" />
                    </ul>
                    <ul className="no-list-style flex-c flex-jc-sb colormap">
                      <li
                        className={`flex-g-1 colormap-0 ${
                          predProb1Active ? 'active' : ''
                        } ${isHovered && !predProb1Active ? 'inactive' : ''}`}
                      >
                        {isHovered && predProb1Active && (
                          <span>
                            {this.state.points[this.hoveredPoint][3].toFixed(2)}
                          </span>
                        )}
                      </li>
                      <li
                        className={`flex-g-1 colormap-1 ${
                          predProb2Active ? 'active' : ''
                        } ${isHovered && !predProb2Active ? 'inactive' : ''}`}
                      >
                        {isHovered && predProb2Active && (
                          <span>
                            {this.state.points[this.hoveredPoint][3].toFixed(2)}
                          </span>
                        )}
                      </li>
                      <li
                        className={`flex-g-1 colormap-2 ${
                          predProb3Active ? 'active' : ''
                        } ${isHovered && !predProb3Active ? 'inactive' : ''}`}
                      >
                        {isHovered && predProb3Active && (
                          <span>
                            {this.state.points[this.hoveredPoint][3].toFixed(2)}
                          </span>
                        )}
                      </li>
                      <li
                        className={`flex-g-1 colormap-3 ${
                          predProb4Active ? 'active' : ''
                        } ${isHovered && !predProb4Active ? 'inactive' : ''}`}
                      >
                        {isHovered && predProb4Active && (
                          <span>
                            {this.state.points[this.hoveredPoint][3].toFixed(2)}
                          </span>
                        )}
                      </li>
                      <li
                        className={`flex-g-1 colormap-4 ${
                          predProb5Active ? 'active' : ''
                        } ${isHovered && !predProb5Active ? 'inactive' : ''}`}
                      >
                        {isHovered && predProb5Active && (
                          <span>
                            {this.state.points[this.hoveredPoint][3].toFixed(2)}
                          </span>
                        )}
                      </li>
                      <li
                        className={`flex-g-1 colormap-5 ${
                          predProb6Active ? 'active' : ''
                        } ${isHovered && !predProb6Active ? 'inactive' : ''}`}
                      >
                        {isHovered && predProb6Active && (
                          <span>
                            {this.state.points[this.hoveredPoint][3].toFixed(2)}
                          </span>
                        )}
                      </li>
                      <li
                        className={`flex-g-1 colormap-6 ${
                          predProb7Active ? 'active' : ''
                        } ${isHovered && !predProb7Active ? 'inactive' : ''}`}
                      >
                        {isHovered && predProb7Active && (
                          <span>
                            {this.state.points[this.hoveredPoint][3].toFixed(2)}
                          </span>
                        )}
                      </li>
                    </ul>
                  </li>
                ) : (
                  <li className="m-t-0-25">
                    <ul className="no-list-style flex-c flex-jc-sb colormap">
                      <li
                        className={`flex-g-1 colormap-positive ${
                          posActive ? 'active' : ''
                        } ${isHovered && !posActive ? 'inactive' : ''}`}
                        title="Positively Labeled Regions"
                      />
                      <li
                        className={`flex-g-1 colormap-negative ${
                          negActive ? 'active' : ''
                        } ${isHovered && !negActive ? 'inactive' : ''}`}
                        title="Negatively Labeled Regions"
                      />
                      <li
                        className={`flex-g-1 colormap-unlabled ${
                          unlabeledActive ? 'active' : ''
                        } ${isHovered && !unlabeledActive ? 'inactive' : ''}`}
                        title="Unlabeled Regions"
                      />
                      <li
                        className={`flex-g-1 colormap-target ${
                          targetActive ? 'active' : ''
                        } ${isHovered && !targetActive ? 'inactive' : ''}`}
                        title="Search Target Regions"
                      />
                    </ul>
                  </li>
                )}
                <li>
                  <LabeledSlider
                    disabled={this.state.isLoading || this.state.isError}
                    id="search-projection-settings-point-size"
                    label="Size"
                    max={10}
                    min={0.5}
                    onChange={this.onChangePointSize}
                    sameLine
                    step={0.5}
                    value={this.state.pointSize}
                  />
                </li>
              </ul>
            )}
          </div>
        </TabEntry>
        <TabEntry
          isOpen={this.props.showProjectionSettings}
          title="Projection Settings"
          toggle={this.props.toggleProjectionSettings}
        >
          <div className="search-right-bar-padding">
            <ul className="no-list-style compact-list">
              <li>
                <LabeledSlider
                  id="search-projection-settings-nn"
                  info="https://umap-learn.readthedocs.io/en/latest/parameters.html#n-neighbors"
                  label="Near. Neigh."
                  max={Math.max(Math.sqrt(this.state.points.length), 5)}
                  min={2}
                  onChange={this.onChangeSettingsUmapNN}
                  step={2}
                  value={this.state.settingsUmapNN}
                />
              </li>
              <li>
                <LabeledSlider
                  id="search-projection-settings-min-dist"
                  info="https://umap-learn.readthedocs.io/en/latest/parameters.html#min-dist"
                  label="Min. Dist."
                  max={1}
                  min={0}
                  onChange={this.onChangeSettingsUmapMinDist}
                  step={0.01}
                  value={this.state.settingsUmapMinDist}
                />
              </li>
              <li>
                <Button onClick={this.newProjection}>Update</Button>
              </li>
            </ul>
          </div>
        </TabEntry>
        <TabEntry
          isOpen={this.props.showProgress}
          title="Training Progress"
          toggle={this.props.toggleProgress}
        >
          <ul className="search-right-bar-padding no-list-style compact-list compact-list-with-padding">
            <li className="flex-c flex-v">
              <div className="bar-chart-x-axis">
                <div className="bar-chart-x-axis-bar">
                  <div className="bar-chart-x-axis-arrow" />
                </div>
                <div className="flex-c flex-jc-c bar-chart-x-axis-title">
                  <span>X Axis: Number of Labels</span>
                </div>
              </div>
              <div className="flex-c flex-jc-sb">
                <span className="label">Uncertainty</span>
                <ButtonIcon
                  onClick={showUncertaintyHelp(this.props.pubSub)}
                  icon="help"
                  iconOnly
                  smaller
                />
              </div>
              <BarChart
                x={this.props.progress.numLabels}
                y={this.props.progress.unpredictabilityAll}
                y2={this.props.progress.unpredictabilityLabels}
                parentWidth={this.props.rightBarWidth}
              />
            </li>
            <li className="flex-c flex-v">
              <div className="flex-c flex-jc-sb">
                <span className="label">
                  Change in the <abbr title="prediction">pred.</abbr>{' '}
                  <abbr title="probability">prob.</abbr>
                </span>
                <ButtonIcon
                  onClick={showPredPropChangeHelp(this.props.pubSub)}
                  icon="help"
                  iconOnly
                  smaller
                />
              </div>
              <BarChart
                x={this.props.progress.numLabels}
                y={this.props.progress.predictionProbaChangeAll}
                y2={this.props.progress.predictionProbaChangeLabels}
                yMax={0.5}
                parentWidth={this.props.rightBarWidth}
              />
            </li>
            <li className="flex-c flex-v">
              <div className="flex-c flex-jc-sb">
                <span className="label">
                  Converge <span className="label-note">(↑)</span> / diverge{' '}
                  <span className="label-note">(↓)</span>
                </span>
                <ButtonIcon
                  onClick={showPredProbDivergenceHelp(this.props.pubSub)}
                  icon="help"
                  iconOnly
                  smaller
                />
              </div>
              <BarChart
                x={this.props.progress.numLabels}
                y={this.props.progress.convergenceAll}
                y2={this.props.progress.convergenceLabels}
                y3={this.props.progress.divergenceAll}
                y4={this.props.progress.divergenceLabels}
                diverging
                parentWidth={this.props.rightBarWidth}
              />
            </li>
            <li className="flex-c flex-jc-sb">
              <span className="label flex-g-1">
                # Labels{' '}
                <span className="label-note">
                  (<abbr title="Positive Labels">Pos</abbr>/
                  <abbr title="Negative Labels">Neg</abbr>)
                </span>
              </span>
              <Badge
                isBordered
                valueA={this.props.searchInfo.classificationsPositive || 0}
                valueB={this.props.searchInfo.classificationsNegative || 0}
              />
            </li>
            <li className="flex-c flex-jc-sb">
              <span className="label flex-g-1"># Trainings</span>
              <Badge
                isBordered
                levelPoor={0}
                levelOkay={1}
                levelGood={3}
                value={this.props.searchInfo.classifiers || 0}
              />
            </li>
          </ul>
        </TabEntry>
        <TabEntry
          isOpen={this.props.showMetadata}
          title="Metadata"
          toggle={this.props.toggleMetadata}
        >
          <div className="search-right-bar-padding">
            <ul className="no-list-style compact-list">
              <li className="flex-c flex-jc-sb">
                <span className="label flex-g-1">Search ID</span>
                <span className="value">{this.props.searchInfo.id || '?'}</span>
              </li>
              <li className="flex-c flex-jc-sb">
                <span className="label flex-g-1">Last Update</span>
                <span className="value">
                  {this.props.searchInfo.updated
                    ? readableDate(this.props.searchInfo.updated)
                    : '?'}
                </span>
              </li>
            </ul>
          </div>
          <div className="flex-c" />
        </TabEntry>
      </div>
    );
  }
}

SearchRightBarInfo.defaultProps = {
  searchInfo: {}
};

SearchRightBarInfo.propTypes = {
  barShow: PropTypes.bool,
  barWidth: PropTypes.number,
  hover: PropTypes.number.isRequired,
  progress: PropTypes.object.isRequired,
  pubSub: PropTypes.object.isRequired,
  rightBarWidth: PropTypes.number.isRequired,
  searchInfo: PropTypes.object,
  selection: PropTypes.array.isRequired,
  setHover: PropTypes.func.isRequired,
  setSelection: PropTypes.func.isRequired,
  showMetadata: PropTypes.bool.isRequired,
  showProgress: PropTypes.bool.isRequired,
  showProjection: PropTypes.bool.isRequired,
  showProjectionSettings: PropTypes.bool,
  tab: PropTypes.oneOfType([PropTypes.string, PropTypes.symbol]),
  toggleMetadata: PropTypes.func.isRequired,
  toggleProgress: PropTypes.func.isRequired,
  toggleProjection: PropTypes.func.isRequired,
  toggleProjectionSettings: PropTypes.func.isRequired
};

const mapStateToProps = state => ({
  barShow: state.present.searchRightBarShow,
  barWidth: state.present.searchRightBarWidth,
  hover: state.present.searchHover,
  rightBarWidth: state.present.searchRightBarWidth,
  selection: state.present.searchSelection,
  showMetadata: state.present.searchRightBarMetadata,
  showProgress: state.present.searchRightBarProgress,
  showProjection: state.present.searchRightBarProjection,
  showProjectionSettings: state.present.searchRightBarProjectionSettings,
  tab: state.present.searchRightBarTab
});

const mapDispatchToProps = dispatch => ({
  setHover: windowId => dispatch(setSearchHover(windowId)),
  setSelection: windowIds => dispatch(setSearchSelection(windowIds)),
  toggleMetadata: isOpen => dispatch(setSearchRightBarMetadata(!isOpen)),
  toggleProgress: isOpen => dispatch(setSearchRightBarProgress(!isOpen)),
  toggleProjection: isOpen => dispatch(setSearchRightBarProjection(!isOpen)),
  toggleProjectionSettings: isOpen =>
    dispatch(setSearchRightBarProjectionSettings(!isOpen))
});

export default connect(mapStateToProps, mapDispatchToProps)(SearchRightBarInfo);
