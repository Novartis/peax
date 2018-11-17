import PropTypes from "prop-types";
import React from "react";
import { connect } from "react-redux";
import { compose } from "recompose";

// Components
import Button from "../components/Button";
import ButtonIcon from "../components/ButtonIcon";
import ButtonRadio from "../components/ButtonRadio";
import ElementWrapperAdvanced from "../components/ElementWrapperAdvanced";
import TabEntry from "../components/TabEntry";

// Factories
import Scatterplot from "../factories/Scatterplot";

// Actions
import {
  setSearchRightBarProjectionCamera,
  setSearchRightBarProjectionSettings,
  setSearchSelection
} from "../actions";

// Utils
import { api, debounce, inputToNum } from "../utils";

// Configs
import {
  BUTTON_RADIO_PROJECTION_COLOR_ENCODING_OPTIONS,
  TAB_RIGHT_BAR_PROJECTION
} from "../configs/search";

// Styles
import "./Search.scss";

const BASE_COLOR = [0.3, 0.3, 0.3, 0.075];
// const HIGHLIGHT_COLOR = [1, 0.749019608, 0, 1];
const COLORMAP_CAT = [
  BASE_COLOR, // Base color
  [0.752941176, 0.141176471, 0.541176471, 0.75], // Negative
  [0.058823529, 0.364705882, 0.57254902, 0.75], // Positive
  [0, 0, 0, 1] // Target
];
const COLORMAP_PRB = [
  // from negative
  [0.85, 0.85, 0.85, 0.05],
  [0.85, 0.85, 0.85, 0.1],
  [0.85, 0.85, 0.85, 0.15],
  [0.8, 0.8, 0.8, 0.2],
  [0.75, 0.75, 0.75, 0.25],
  [0.7, 0.7, 0.7, 0.3],
  [0.65, 0.65, 0.65, 0.35],
  [0.6, 0.6, 0.6, 0.4],
  // neutral
  [0.5, 0.5, 0.5, 0.5],
  [0.47058823529411764, 0.48627450980392156, 0.5215686274509804, 0.6],
  [0.43529411764705883, 0.4745098039215686, 0.5411764705882353, 0.7],
  [0.4, 0.4588235294117647, 0.5607843137254902, 0.75],
  [0.3568627450980392, 0.44313725490196076, 0.5803921568627451, 0.75],
  [0.30980392156862746, 0.43137254901960786, 0.6, 0.75],
  [0.25098039215686274, 0.4196078431372549, 0.6196078431372549, 0.8],
  [0.17254901960784313, 0.403921568627451, 0.6392156862745098, 0.85],
  [0, 0.39215686274509803, 0.6588235294117647, 1]
  // to positive
];
const REDRAW_DELAY = 750;
const PROJECTION_CHECK_INTERVAL = 2000;
const PROJECTION_CAMERA_POSITION = [0, 0, 1];
const PROJECTION_CAMERA_INTERVAL = 500;

class SearchRightBarProjection extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      canvas: null,
      classes: [],
      colorEncoding: "categorical",
      isColorByProb: false,
      isError: false,
      isInit: false,
      isLoading: false,
      pointSize: 3,
      probabilities: [],
      projection: [],
      projectionPrepared: [],
      projectionPrepToOrig: [],
      scatterplot: null,
      selected: [],
      settingsUmapMinDist: 0.1,
      settingsUmapNN: 5
    };

    this.onChangeColorEncoding = compose(this.onChangeState("colorEncoding"));
    this.onChangePointSize = compose(
      this.onChangeState("pointSize"),
      pointSize => {
        if (this.scatterplot) {
          // Progresively increase the step size
          const n = Math.floor((pointSize - 1) / 3);
          const prevExtraSteps = ((n * (n - 1)) / 2) * 3;
          const stepSize = Math.ceil(pointSize / 3);
          const extraStep = stepSize - 1;
          const finalPtSize =
            pointSize +
            prevExtraSteps +
            extraStep * (pointSize - extraStep * 3);
          this.scatterplot.pointSize = finalPtSize;
          this.drawScatterplot();
        }
        return pointSize;
      },
      inputToNum
    );
    this.onChangeSettingsUmapNN = compose(
      this.onChangeState("settingsUmapNN"),
      inputToNum
    );
    this.onChangeSettingsUmapMinDist = compose(
      this.onChangeState("settingsUmapMinDist"),
      inputToNum
    );
    this.onRefBnd = this.onRef.bind(this);
    this.newProjectionBnd = this.newProjection.bind(this);
    this.onResetLocationBnd = this.onResetLocation.bind(this);
    this.onClickBnd = this.onClick.bind(this);

    this.onCameraChangeDb = debounce(
      this.onCameraChange.bind(this),
      PROJECTION_CAMERA_INTERVAL
    );
    this.drawScatterplotDb = debounce(
      this.drawScatterplot.bind(this),
      REDRAW_DELAY
    );
  }

  componentDidMount() {
    if (this.props.searchInfo.id) this.loadProjection();
    this.checkCameraPos();
  }

  componentDidUpdate(prevProps, prevState) {
    if (this.isOpen && this.props.searchInfo.id && !this.state.isInit)
      this.loadProjection();
    if (this.isOpen && this.props.tab !== prevProps.tab)
      this.drawScatterplotDb(true);
    if (
      this.isOpen &&
      this.state.projectionPrepared !== prevState.projectionPrepared
    )
      this.drawScatterplot(true);
    if (
      this.state.colorEncoding !== prevState.colorEncoding ||
      this.props.selection !== prevProps.selection
    ) {
      const {
        projectionPrepared,
        projectionPrepToOrig
      } = this.prepareProjection();
      this.setState({ projectionPrepared, projectionPrepToOrig });
    }
    if (this.props.barWidth && this.props.barWidth !== prevProps.barWidth)
      this.drawScatterplotDb(true);
    if (this.props.projectionCamera !== prevProps.projectionCamera)
      this.checkCameraPos();
  }

  componentWillUnmount() {
    if (this.scatterplot) {
      this.scatterplot.unsubscribe("camera", this.onResetLocationDb);
      this.scatterplot.unsubscribe("click", this.onClickBnd);
    }
  }

  /* ---------------------------- Custom Methods ---------------------------- */

  get isOpen() {
    return this.props.barShow && this.props.tab === TAB_RIGHT_BAR_PROJECTION;
  }

  drawScatterplot(updateSize = false) {
    if (!this.scatterplot) this.initScatterplot();
    else this.updateScatterplot(this.state.projectionPrepared, updateSize);
  }

  initScatterplot(projection = this.state.projectionPrepared) {
    if (!projection.length || !this.canvasWrapper) return;

    const bBox = this.canvasWrapper.getBoundingClientRect();

    const scatterplot = Scatterplot({
      width: bBox.width,
      height: bBox.height,
      padding: 0,
      pointWidth: this.state.pointSize
    });

    scatterplot.subscribe("camera", this.onCameraChangeDb);
    scatterplot.subscribe("click", this.onClickBnd);

    scatterplot.draw(projection, this.props.selection.length);

    this.scatterplot = scatterplot;

    this.setState({ canvas: scatterplot.canvas });
  }

  updateScatterplot(
    projection = this.state.projectionPrepared,
    updateSize = false
  ) {
    if (!projection.length || !this.canvasWrapper) return;
    if (updateSize) {
      const bBox = this.canvasWrapper.getBoundingClientRect();
      this.scatterplot.width = bBox.width;
      this.scatterplot.height = bBox.height;
      this.scatterplot.refresh();
    }
    this.scatterplot.draw(projection, this.props.selection.length);
  }

  prepareProjection(
    projection = this.state.projection,
    classes = this.state.classes,
    probabilities = this.state.probabilities
  ) {
    const numProbColors = COLORMAP_PRB.length;
    // Lists for the points
    const projectionPrepared = [];
    const neg = [];
    const pos = [];
    const tar = [];
    const sel = [];
    // Lists for the index mapping
    const projectionPrepToOrig = [];
    const negIdx = [];
    const posIdx = [];
    const tarIdx = [];
    const selIdx = [];

    for (let i = 0; i < projection.length; i += 2) {
      const j = i / 2;
      const color =
        this.state.colorEncoding === "probability"
          ? COLORMAP_PRB[Math.round(probabilities[j] * numProbColors)].slice()
          : COLORMAP_CAT[classes[j]].slice();

      let list = projectionPrepared;
      let listIdx = projectionPrepToOrig;
      let pointSize = 0;
      switch (classes[j]) {
        case 1:
          list = neg;
          listIdx = negIdx;
          pointSize = 1;
          break;

        case 2:
          list = pos;
          listIdx = posIdx;
          pointSize = 2;
          break;

        case 3:
          list = tar;
          listIdx = tarIdx;
          pointSize = 4;
          break;

        default:
        // Nothing
      }

      if (this.props.selection.indexOf(j) >= 0) {
        list = sel;
        listIdx = selIdx;
      }

      list.push([projection[i], projection[i + 1], [...color], pointSize]);
      listIdx.push(j);
    }

    projectionPrepared.push(...neg, ...pos, ...tar, ...sel);
    projectionPrepToOrig.push(...negIdx, ...posIdx, ...tarIdx, ...selIdx);

    return { projectionPrepared, projectionPrepToOrig };
  }

  async newProjection() {
    if (this.state.isLoading) return;

    this.setState({ isLoading: true, isError: false, isNotFound: false });

    const resp = await api.newProjection(this.props.searchInfo.id);
    const isError = resp.status !== 200 ? "Couldn't project data." : false;

    const checkProjectionTimer = isError
      ? null
      : setInterval(this.checkProjection.bind(this), PROJECTION_CHECK_INTERVAL);

    this.setState({
      isError,
      checkProjectionTimer
    });
  }

  async checkProjection() {
    const resp = await api.newProjection(this.props.searchInfo.id);
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

    // Compare the number of windows for which we got classifications, projections, and
    // projections. Those should be the same otherwise the data seems to be corrupted
    const numDiffLenghts = new Set([
      respClasses.body.results.length,
      respProbs.body.results.length,
      respProj.body.projection.length / 2
    ]).size;

    const isNotFound =
      respProj.status === 404 ? "Projection not computed." : false;

    let isError =
      !isNotFound && numDiffLenghts > 1 ? "Data is correpted! RUN!1!" : false;

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
    const { projectionPrepared, projectionPrepToOrig } = this.prepareProjection(
      projection,
      classes,
      probabilities
    );

    this.setState({
      isNotFound,
      isError,
      isInit: true,
      isLoading: false,
      classes,
      probabilities,
      projection,
      projectionPrepared,
      projectionPrepToOrig
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

  onRef(canvasWrapper) {
    this.canvasWrapper = canvasWrapper;
  }

  onCameraChange(camera) {
    this.props.setProjectionCamera(camera);
  }

  onClick({ selectedPoint }) {
    this.props.setSelection(
      +selectedPoint >= 0
        ? [this.state.projectionPrepToOrig[selectedPoint]]
        : []
    );
  }

  checkCameraPos() {
    const isCameraDefaultPos = this.props.projectionCamera.every(
      (x, i) =>
        Math.round(x * 1000000) === PROJECTION_CAMERA_POSITION[i] * 1000000
    );
    this.setState({ isCameraDefaultPos });
  }

  onResetLocation() {
    this.scatterplot.reset();
  }

  /* -------------------------------- Render -------------------------------- */

  render() {
    return (
      <div className="right-bar-info flex-c flex-v full-wh">
        <div className="search-right-bar-padding">
          <div className="search-projection-wrapper" ref={this.onRefBnd}>
            {this.isOpen && (
              <ElementWrapperAdvanced
                className="search-projection"
                element={this.state.canvas}
                isError={this.state.isError}
                isLoading={this.state.isLoading}
                isNotFound={this.state.isNotFound}
              />
            )}
            {!this.state.isCameraDefaultPos && (
              <ButtonIcon
                className="search-projection-reset"
                icon="reset"
                iconOnly={true}
                isIconRotationOnFocus={true}
                isDisabled={this.state.isCameraDefaultPos}
                onClick={this.onResetLocationBnd}
              />
            )}
          </div>
          {this.state.isNotFound && (
            <ul className="no-list-style compact-list right-bar-v-padding">
              <li className="flex-c flex-jc-sb">
                <Button onClick={this.newProjectionBnd}>
                  Compute projection
                </Button>
              </li>
            </ul>
          )}
          {!!this.state.projection.length && (
            <ul className="no-list-style compact-list right-bar-v-padding">
              <li className="flex-c flex-jc-sb">
                <ButtonRadio
                  className="full-w"
                  name="search-projection-color-encoding"
                  onClick={this.onChangeColorEncoding}
                  options={BUTTON_RADIO_PROJECTION_COLOR_ENCODING_OPTIONS}
                  selection={this.state.colorEncoding}
                />
              </li>
              <li>
                <label
                  className="flex-c"
                  htmlFor="search-projection-settings-point-size"
                >
                  Point size
                </label>
                <input
                  id="search-projection-settings-point-size"
                  type="range"
                  min="2"
                  max="12"
                  step="1"
                  disabled={this.state.isLoading || this.state.isError}
                  value={this.state.pointSize}
                  onChange={this.onChangePointSize}
                />
              </li>
            </ul>
          )}
        </div>
        <TabEntry
          isOpen={this.props.settingsIsOpen}
          title="Settings"
          toggle={this.props.toggleSettings}
        >
          <div className="search-right-bar-padding">
            <ul className="no-list-style compact-list">
              <li>
                <label
                  className="flex-c flex-jc-sb"
                  htmlFor="search-projection-settings-nn"
                >
                  <abbr title="Nearest Neighbors">Near. Neigh.</abbr>
                  <ButtonIcon
                    className="info-external"
                    external={true}
                    icon="info"
                    iconOnly={true}
                    href="https://umap-learn.readthedocs.io/en/latest/parameters.html#n-neighbors"
                  />
                </label>
                <input
                  className="full-w"
                  id="search-projection-settings-nn"
                  type="number"
                  min="2"
                  step="1"
                  value={this.state.settingsUmapNN}
                  onChange={this.onChangeSettingsUmapNN}
                  disabled={true}
                />
              </li>
              <li>
                <label
                  className="flex-c flex-jc-sb"
                  htmlFor="search-projection-settings-min-dist"
                >
                  <abbr title="Minimum Distance">Min. Dist.</abbr>
                  <ButtonIcon
                    className="info-external"
                    external={true}
                    icon="info"
                    iconOnly={true}
                    href="https://umap-learn.readthedocs.io/en/latest/parameters.html#min-dist"
                  />
                </label>
                <input
                  className="full-w"
                  id="search-projection-settings-min-dist"
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={this.state.settingsUmapMinDist}
                  onChange={this.onChangeSettingsUmapMinDist}
                  disabled={true}
                />
              </li>
              <li>
                <Button onClick={this.newProjectionBnd}>Update</Button>
              </li>
            </ul>
          </div>
        </TabEntry>
      </div>
    );
  }
}

SearchRightBarProjection.defaultProps = {
  searchInfo: {}
};

SearchRightBarProjection.propTypes = {
  barShow: PropTypes.bool,
  barWidth: PropTypes.number,
  hoveringWindowId: PropTypes.number,
  projectionCamera: PropTypes.array,
  searchInfo: PropTypes.object,
  selection: PropTypes.arrayOf(PropTypes.number),
  setProjectionCamera: PropTypes.func,
  setSelection: PropTypes.func,
  settingsIsOpen: PropTypes.bool,
  tab: PropTypes.oneOfType([PropTypes.string, PropTypes.symbol]),
  toggleSettings: PropTypes.func
};

const mapStateToProps = state => ({
  barShow: state.present.searchRightBarShow,
  barWidth: state.present.searchRightBarWidth,
  projectionCamera: state.present.searchRightBarProjectionCamera,
  selection: state.present.searchSelection,
  settingsIsOpen: state.present.searchRightBarProjectionSettings,
  tab: state.present.searchRightBarTab
});

const mapDispatchToProps = dispatch => ({
  setProjectionCamera: camera =>
    dispatch(setSearchRightBarProjectionCamera(camera)),
  setSelection: windowId => dispatch(setSearchSelection(windowId)),
  toggleSettings: isOpen =>
    dispatch(setSearchRightBarProjectionSettings(!isOpen))
});

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(SearchRightBarProjection);
