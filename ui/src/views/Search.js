import { boundMethod } from 'autobind-decorator';
import update from 'immutability-helper';
import PropTypes from 'prop-types';
import React from 'react';
import { connect } from 'react-redux';
import { Link, withRouter } from 'react-router-dom';
import { compose } from 'recompose';
import { ChromosomeInfo } from 'higlass';

// Higher-order components
import { withPubSub } from '../hocs/pub-sub';

// Components
import Content from '../components/Content';
import ContentWrapper from '../components/ContentWrapper';
import HiGlassViewer from '../components/HiGlassViewer';
import ErrorMsgCenter from '../components/ErrorMsgCenter';
import SpinnerCenter from '../components/SpinnerCenter';
import TabContent from '../components/TabContent';
import ToolTip from '../components/ToolTip';

// View components
import NotFound from './NotFound';
import SearchClassifications from './SearchClassifications';
import SearchResults from './SearchResults';
import SearchRightBar from './SearchRightBar';
import SearchSeeds from './SearchSeeds';
import SearchSelection from './SearchSelection';
import SearchSubTopBar from './SearchSubTopBar';
import SearchSubTopBarAll from './SearchSubTopBarAll';
import SearchSubTopBarTabs from './SearchSubTopBarTabs';

// Actions
import {
  setSearchHover,
  setSearchRightBarShow,
  setSearchRightBarTab,
  setSearchTab
} from '../actions';

// Utils
import {
  api,
  classifToNum,
  Deferred,
  inputToNum,
  Logger,
  numToClassif,
  readableDate,
  removeHiGlassEventListeners,
  requestNextAnimationFrame
} from '../utils';

// Configs
import {
  PROGRESS_CHECK_INTERVAL,
  TAB_CLASSIFICATIONS,
  TAB_RESULTS,
  TAB_RIGHT_BAR_HELP,
  TAB_RIGHT_BAR_INFO,
  TAB_SEEDS,
  TAB_SELECTION,
  TRAINING_CHECK_INTERVAL,
  PER_PAGE_ITEMS
} from '../configs/search';

const logger = Logger('Search');

const baseUrl = 'http://localhost:5000/api/v1';

const resizeTrigger = () =>
  requestNextAnimationFrame(() => {
    window.dispatchEvent(new Event('resize'));
  });

const showInfo = (pubSub, msg) => {
  pubSub.publish('globalDialog', {
    message: msg,
    request: new Deferred(),
    resolveOnly: true,
    resolveText: 'Okay',
    headline: 'Peax'
  });
};

class Search extends React.Component {
  constructor(props) {
    super(props);

    this.hiGlassEventListeners = [];
    this.pubSubs = [];

    this.state = {
      classifications: [],
      dataTracks: [],
      info: {},
      isComputingProgress: false,
      isError: false,
      isErrorClassifications: false,
      isErrorProgress: false,
      isErrorResults: false,
      isErrorSeeds: false,
      isErrorSelection: false,
      isInit: true,
      isLoading: false,
      isLoadingClassifications: false,
      isLoadingResults: false,
      isLoadingSeeds: false,
      isLoadingSelection: false,
      isLoadingProgress: false,
      isMinMaxValuesByTarget: false,
      isTraining: null,
      locationEnd: null,
      locationStart: null,
      minMaxSource: null,
      minMaxValues: {},
      pageClassifications: 0,
      pageResults: 0,
      pageSeeds: 0,
      pageSelection: 0,
      progress: {},
      results: [],
      resultsConflictsFp: [],
      resultsConflictsFn: [],
      resultsProbs: [],
      resultsPredictionHistogram: [],
      resultsPredictionProbBorder: null,
      searchInfo: null,
      searchInfosAll: null,
      seeds: {},
      selection: [],
      windows: {},
      predictionProbBorder: 0.5
    };

    this.onPageClassifications = this.onPage('classifications');
    this.onPageResults = this.onPage('results');
    this.onPageSeeds = this.onPage('seeds');
    this.onPageSelection = this.onPage('selection');

    this.pubSubs.push(
      this.props.pubSub.subscribe('keydown', this.keyDownHandler)
    );
    this.pubSubs.push(this.props.pubSub.subscribe('keyup', this.keyUpHandler));

    this.onChangePreditionProbBorder = compose(
      this.onChangeState('predictionProbBorder'),
      inputToNum
    );

    this.viewConfigAdjustmentsSelections = {
      key: 'views.0.tracks.top.0.contents.1.options.regions',
      value: []
    };
    this.viewConfigAdjustmentsLabels = {
      key: 'views.0.tracks.top.1.contents.1.options.regions',
      value: []
    };
    this.viewConfigAdjustments = [
      this.viewConfigAdjustmentsSelection,
      this.viewConfigAdjustmentsLabels
    ];
  }

  componentDidMount() {
    this.loadMetadata();
    this.loadClassifications();
    this.loadProgress();
  }

  componentWillUnmount() {
    this.pubSubs.forEach(subscription =>
      this.props.pubSub.unsubscribe(subscription)
    );
    this.pubSubs = [];
    removeHiGlassEventListeners(this.hiGlassEventListeners, this.hgApi);
    this.hiGlassEventListeners = [];
  }

  shouldComponentUpdate(nextProps, nextState) {
    if (
      nextState.searchInfo !== this.state.searchInfo ||
      nextProps.selectedRegions !== this.props.selectedRegions
    ) {
      this.translateSelectedRegions(
        nextState.searchInfo,
        nextProps.selectedRegions
      );
    }
    if (
      nextState.searchInfo !== this.state.searchInfo ||
      nextState.windows !== this.state.windows
    ) {
      this.translateLabeledRegions(nextState.searchInfo, nextState.windows);
    }
    return true;
  }

  componentDidUpdate(prevProps, prevState) {
    if (this.id !== prevProps.match.params.id) {
      this.loadMetadata();
      this.loadProgress();
    }
    if (this.props.tab !== prevProps.tab) this.checkTabData();
    if (this.state.minMaxValues !== prevState.minMaxValues) {
      this.normalize();
    }
    if (
      (this.props.tab !== prevProps.tab ||
        this.state.pageClassifications !== prevState.pageClassifications ||
        this.state.pageResults !== prevState.pageResults ||
        this.state.pageSeeds !== prevState.pageSeeds) &&
      !this.state.isMinMaxValuesByTarget
    ) {
      this.denormalize();
    }
    if (
      prevProps.selectedRegions !== this.props.selectedRegions &&
      this.props.tab === TAB_SELECTION
    ) {
      this.prepareSelection();
    }
  }

  /* -------------------------- Getters & Setters --------------------------- */

  get id() {
    return this.props.match.params.id;
  }

  get isSeeded() {
    if (!this.state.searchInfo) return false;
    return (
      this.state.searchInfo.classifications >=
      this.state.info.minClassifications
    );
  }

  get isTrained() {
    return (
      this.state.searchInfo &&
      this.state.searchInfo.classifiers > 0 &&
      this.state.isTraining === false
    );
  }

  /* ---------------------------- Custom Methods ---------------------------- */

  checkTabData() {
    if (!this.state.searchInfo || this.state.searchInfo.status === 404) return;

    if (this.props.tab === TAB_SEEDS && !Object.values(this.state.seeds).length)
      this.loadSeeds();

    if (this.props.tab === TAB_RESULTS) this.loadResults();

    if (this.props.tab === TAB_CLASSIFICATIONS) this.loadClassifications();

    if (this.props.tab === TAB_SELECTION) this.prepareSelection();
  }

  async loadMetadata() {
    if (this.state.isLoading && !this.state.isInit) return;

    this.setState({ isLoading: true, isError: false });

    const info = await api.getInfo();

    let searchInfo;
    let searchInfosAll;

    let dataTracks = await api.getDataTracks();
    let isError =
      dataTracks.status !== 200 ? "Couldn't load data tracks" : false;
    dataTracks = isError ? null : dataTracks.body.results;

    if (typeof this.id !== 'undefined') {
      searchInfo = await api.getSearchInfo(this.id);
      isError =
        searchInfo.status !== 200 ? "Couldn't load search info." : false;
      if (!isError) {
        searchInfo = searchInfo.body;
        searchInfo.classificationsPositive =
          searchInfo.classifications_positive;
        searchInfo.classificationsNegative =
          searchInfo.classifications - searchInfo.classifications_positive;
        this.chromInfo = ChromosomeInfo(
          `${baseUrl}/chrom-sizes/?id=${searchInfo.coords}&type=csv`
        );
      } else {
        searchInfo = null;
      }
    } else {
      searchInfosAll = await api.getAllSearchInfos();
      isError =
        searchInfosAll.status !== 200 ? "Couldn't load search infos." : false;
      searchInfosAll = isError ? null : searchInfosAll.body;
    }

    this.setState({
      dataTracks,
      info,
      isError,
      isInit: false,
      isLoading: false,
      searchInfo,
      searchInfosAll
    });

    this.checkTabData();
  }

  async loadSeeds() {
    if (this.state.isLoadingSeeds) return;

    this.setState({ isLoadingSeeds: true, isErrorSeeds: false });

    const seedsInfo = await api.getSeeds(this.id);
    const isErrorSeeds =
      seedsInfo.status !== 200 ? "Couldn't load seeds." : false;
    const seeds =
      !isErrorSeeds && seedsInfo.body.results ? seedsInfo.body.results : [];
    const training =
      !isErrorSeeds && !seedsInfo.body.results ? seedsInfo.body : undefined;

    // Check if seeds are ready or wether the classifier is still training
    if (seeds) {
      // Seeds are ready! Lets go and visualize them
      const newSeeds = seeds.reduce((a, b) => {
        a[b] = true;
        return a;
      }, {});

      this.setState({
        isLoadingSeeds: false,
        isErrorSeeds,
        seeds: newSeeds
      });
    }
    if (training) {
      // Classifier is training
      const trainingCheckTimerId = training.isTraining
        ? setInterval(this.onTrainingCheckSeeds, TRAINING_CHECK_INTERVAL)
        : null;

      if (training.isTraining) this.onTrainingCheckSeeds();

      this.setState({
        isTraining: training.isTraining,
        trainingCheckTimerId
      });
    }
  }

  async loadClassifications() {
    if (this.state.isLoadingClassifications) return;

    this.setState({
      isLoadingClassifications: true,
      isErrorClassifications: false
    });

    let classifications = await api.getClassifications(this.id);
    const isErrorClassifications =
      classifications.status !== 200 ? "Could't load classifications." : false;
    classifications = isErrorClassifications
      ? []
      : classifications.body.results;

    const windows = { ...this.state.windows };
    classifications.forEach(({ windowId, classification }) => {
      windows[windowId] = {
        classification: numToClassif(classification),
        classificationNumerical: classification,
        classificationPending: false
      };
    });

    this.setState({
      isLoadingClassifications: false,
      isErrorClassifications,
      classifications,
      windows
    });
  }

  async prepareSelection() {
    if (this.state.isLoadingClassifications) return;

    this.setState({
      isLoadingSelection: true,
      isErrorSelection: false
    });

    await this.loadClassifications();
    await this.loadResults();

    const index = this.state.results.reduce((idx, win) => {
      idx[win.windowId] = {
        classification: win.classification,
        probability: win.probability
      };
      return idx;
    }, {});

    this.state.resultsConflictsFn.reduce((idx, fn) => {
      idx[fn.windowId] = {
        classification: fn.classification,
        probability: fn.probability,
        conflict: 'fn'
      };
      return idx;
    }, index);

    this.state.resultsConflictsFp.reduce((idx, fp) => {
      idx[fp.windowId] = {
        classification: fp.classification,
        probability: fp.probability,
        conflict: 'fp'
      };
      return idx;
    }, index);

    const isErrorSelection = this.state.isErrorClassifications
      ? "Could't load classifications."
      : false;

    const selection = isErrorSelection
      ? []
      : this.props.selectedRegions.map(windowId => ({
          classification:
            typeof this.state.windows[windowId] !== 'undefined'
              ? this.state.windows[windowId].classificationNumerical
              : 0,
          probability:
            typeof index[windowId] !== 'undefined'
              ? index[windowId].probability
              : undefined,
          conflict:
            typeof index[windowId] !== 'undefined'
              ? index[windowId].conflict
              : undefined,
          windowId
        }));

    this.setState({
      isLoadingSelection: false,
      isErrorSelection,
      selection,
      pageSelection: 0
    });
  }

  @boundMethod
  async loadResults() {
    if (this.state.isLoadingResults) return;

    this.setState({
      isLoadingResults:
        this.isSeeded &&
        this.state.searchInfo.classifiers > 0 &&
        !this.state.isTraining,
      isErrorResults: false
    });

    if (
      this.state.searchInfo.classifiers > 0 &&
      this.state.isTraining === null
    ) {
      let trainingInfo = await api.getClassifier(this.id);
      const isErrorResults =
        trainingInfo.status !== 200 ? "Could't load classifier info." : false;
      trainingInfo = isErrorResults ? {} : trainingInfo.body;

      await this.setState({
        isErrorResults,
        isTraining: trainingInfo.isTraining
      });
    }

    if (this.isSeeded && this.isTrained) {
      let predictions = await api.getPredictions(
        this.id,
        this.state.predictionProbBorder
      );
      const isErrorResults =
        predictions.status !== 200 ? "Could't load results." : false;
      predictions = isErrorResults ? [] : predictions.body;

      this.setState({
        isLoadingResults: false,
        isErrorResults,
        results: predictions.results,
        resultsConflictsFp: predictions.conflictsFp,
        resultsConflictsFn: predictions.conflictsFn,
        resultsPredictionHistogram: predictions.predictionHistogram,
        resultsPredictionProbBorder: predictions.predictionProbBorder
      });
    }
  }

  async loadProgress() {
    if (!this.id || this.state.isLoadingProgress) return;

    this.setState({ isLoadingProgress: true, isErrorProgress: false });

    const progressInfo = await api.getProgress(this.id);
    const isErrorProgress =
      progressInfo.status !== 200 ? "Couldn't load seeds." : false;
    const isComputing = !isErrorProgress && progressInfo.body.isComputing;
    const progress =
      !isErrorProgress && !isComputing ? progressInfo.body : null;

    if (isErrorProgress) {
      this.setState({
        isLoadingProgress: false,
        isErrorProgress
      });
    } else if (isComputing) {
      // Classifier is training
      const progressCheckTimerId = setInterval(
        this.onProgressCheck,
        PROGRESS_CHECK_INTERVAL
      );

      this.setState({
        isComputingProgress: true,
        progressCheckTimerId
      });
    } else {
      // Check if progress is available
      this.setState({
        isLoadingProgress: false,
        isErrorProgress: false,
        progress: {
          unpredictabilityAll: progress.unpredictabilityAll,
          unpredictabilityLabels: progress.unpredictabilityLabels,
          predictionProbaChangeAll: progress.predictionProbaChangeAll,
          predictionProbaChangeLabels: progress.predictionProbaChangeLabels,
          convergenceAll: progress.convergenceAll,
          convergenceLabels: progress.convergenceLabels,
          divergenceAll: progress.divergenceAll,
          divergenceLabels: progress.divergenceLabels,
          numLabels: progress.numLabels
        }
      });
    }
  }

  translateLabeledRegions(
    searchInfo = this.state.searchInfo,
    windows = this.state.windows
  ) {
    if (!searchInfo) return;

    const stepSize = searchInfo.windowSize / searchInfo.config.step_freq;

    this.viewConfigAdjustmentsLabels.value = Object.keys(
      windows
    ).map(windowId => [
      searchInfo.dataFrom + windowId * stepSize,
      searchInfo.dataFrom + windowId * stepSize + searchInfo.windowSize,
      windows[windowId].classification[0] === 'n' ? '#cc168c' : '#0f5d92'
    ]);

    this.viewConfigAdjustments = [
      this.viewConfigAdjustmentsSelections,
      this.viewConfigAdjustmentsLabels
    ];
  }

  translateSelectedRegions(
    searchInfo = this.state.searchInfo,
    selectedRegions = this.props.selectedRegions
  ) {
    if (!searchInfo) return;

    const stepSize = searchInfo.windowSize / searchInfo.config.step_freq;

    this.viewConfigAdjustmentsSelections.value = selectedRegions.map(
      regionId => [
        searchInfo.dataFrom + regionId * stepSize,
        searchInfo.dataFrom + regionId * stepSize + searchInfo.windowSize
      ]
    );

    this.viewConfigAdjustments = [
      this.viewConfigAdjustmentsSelections,
      this.viewConfigAdjustmentsLabels
    ];
  }

  /**
   * Wrapper for triggering a public function of HiGlass
   *
   * @description
   * We need an extra wrapper because the HiGlass's might not be available by
   * the time we pass props to a component.
   *
   * @param  {String}  method  Function name to be triggered.
   * @return  {Function}  Curried function calling the HiGlass API.
   */
  callHgApi(method) {
    return (...args) => {
      if (!this.hgApi) {
        logger.warn('HiGlass not available yet.');
        return undefined;
      }
      if (!this.hgApi[method]) {
        logger.warn(
          `Method (${method}) not available. Incompatible version of HiGlass?`
        );
        return undefined;
      }
      return this.hgApi[method](...args);
    };
  }

  @boundMethod
  checkHgApi(newHgApi) {
    if (this.hgApi !== newHgApi) {
      removeHiGlassEventListeners(this.hiGlassEventListeners, this.hgApi);
      this.hiGlassEventListeners = {};

      this.hgApi = newHgApi;

      this.checkHgEvents();
    }
  }

  checkHgEvents() {
    if (!this.hgApi) return;

    if (!this.hiGlassEventListeners.location) {
      this.hiGlassEventListeners.location = {
        name: 'location',
        id: this.hgApi.on('location', this.locationHandler)
      };
    }

    if (!this.hiGlassEventListeners.cursorLocation) {
      this.hiGlassEventListeners.cursorLocation = {
        name: 'cursorLocation',
        id: this.hgApi.on('cursorLocation', this.cursorLocationHandler)
      };
    }
  }

  @boundMethod
  resetViewport() {
    this.setState({ viewportChanged: false });
    this.callHgApi('resetViewport')();
  }

  @boundMethod
  locationHandler({ xDomain }) {
    if (this.state.locationStart === null) {
      this.setState({
        locationStart: xDomain[0],
        locationEnd: xDomain[1]
      });
    } else if (
      xDomain[0] !== this.state.locationStart ||
      xDomain[1] !== this.state.locationEnd
    ) {
      this.setState({ viewportChanged: true });
    }
  }

  @boundMethod
  cursorLocationHandler({ dataX }) {
    if (
      !this.state.searchInfo ||
      dataX < this.state.searchInfo.dataFrom ||
      dataX >= this.state.searchInfo.dataTo
    )
      return;

    const stepSize =
      this.state.searchInfo.windowSize / this.state.searchInfo.config.step_freq;

    const windowId = Math.floor(
      (dataX - this.state.searchInfo.dataFrom) / stepSize
    );

    this.props.setHover(windowId);
  }

  removeHiGlassEventListeners() {
    this.hiGlassEventListeners.forEach(event => {
      this.hgApi.off(event.name, event.id);
    });
    this.hiGlassEventListeners = [];
  }

  @boundMethod
  keyDownHandler(event) {
    if (event.keyCode === 83) {
      // S
      event.preventDefault();

      if (event.ctrlKey || event.metaKey) {
        // CMD + S
        logger.warn('Not implemented yet.');
      }
    }
  }

  @boundMethod
  keyUpHandler(event) {
    if (event.keyCode === 73) {
      // I
      event.preventDefault();
      if (this.props.rightBarTab !== TAB_RIGHT_BAR_INFO) {
        this.props.setRightBarTab(TAB_RIGHT_BAR_INFO);
        if (!this.props.rightBarShow) {
          this.props.setRightBarShow(true);
        }
      } else if (!this.props.rightBarShow) {
        this.props.setRightBarShow(true);
      } else {
        this.props.setRightBarShow(false);
      }
    }

    if (event.keyCode === 72) {
      // H
      event.preventDefault();
      if (this.props.rightBarTab !== TAB_RIGHT_BAR_HELP) {
        this.props.setRightBarTab(TAB_RIGHT_BAR_HELP);
        if (!this.props.rightBarShow) {
          this.props.setRightBarShow(true);
        }
      } else if (!this.props.rightBarShow) {
        this.props.setRightBarShow(true);
      } else {
        this.props.setRightBarShow(false);
      }
    }

    if (event.keyCode === 82) {
      // R
      event.preventDefault();
      this.resetViewport();
    }
  }

  @boundMethod
  resetAllViewports() {
    logger.warn('Sorry, `Search.resetAllViewports()` not implemented yet.');
  }

  @boundMethod
  classificationChangeHandler(windowId) {
    return async classif => {
      const isNew = !this.state.windows[windowId];
      const oldClassif = !isNew && this.state.windows[windowId].classification;

      if (
        (!isNew && this.state.windows[windowId].classificationPending) ||
        (isNew && classif === 'neutral')
      )
        return;

      // Optimistic update
      this.setState({
        windows: update(this.state.windows, {
          [windowId]: win =>
            update(win || {}, {
              classification: { $set: classif },
              classificationNumerical: { $set: classifToNum(classif) },
              classificationPending: { $set: true }
            })
        })
      });

      const setNewClassif = classif === 'positive' || classif === 'negative';

      // Send the new classification back to the server
      const response = setNewClassif
        ? await api.setClassification(this.id, windowId, classif)
        : await api.deleteClassification(this.id, windowId, classif);

      // Set confirmed classification
      this.setState({
        windows: update(this.state.windows, {
          [windowId]: {
            classification: {
              $set: response.status === 200 ? classif : oldClassif
            },
            classificationNumerical: {
              $set: classifToNum(response.status === 200 ? classif : oldClassif)
            },
            classificationPending: { $set: false }
          }
        })
      });

      const numNewClassif = setNewClassif ? 1 : -1;

      this.setState({
        searchInfo: update(this.state.searchInfo, {
          classifications: {
            $set: this.state.searchInfo.classifications + numNewClassif
          }
        })
      });
    };
  }

  async denormalize() {
    if (!this.hgApi) return;

    const minMaxValues = {};
    this.state.dataTracks.forEach(track => {
      minMaxValues[track] = [undefined, undefined];
    });

    this.setState({
      minMaxSource: undefined,
      minMaxValues,
      isMinMaxValuesByTarget: false
    });
  }

  async normalize() {
    if (!this.hgApi) return;

    if (
      this.state.minMaxSource !== 'target' &&
      this.state.isMinMaxValuesByTarget
    ) {
      await this.setState({ isMinMaxValuesByTarget: false });
    }

    Object.keys(this.state.minMaxValues).forEach(track => {
      this.hgApi.setTrackValueScaleLimits(
        undefined,
        track,
        ...this.state.minMaxValues[track]
      );
    });
  }

  @boundMethod
  normalizeByTarget() {
    if (!this.hgApi) return;

    const minMaxValues = {};
    this.state.dataTracks.forEach(track => {
      if (this.state.isMinMaxValuesByTarget) {
        minMaxValues[track] = [undefined, undefined];
      } else {
        minMaxValues[track] = [
          0,
          this.hgApi.getMinMaxValue(undefined, track, true)[1]
        ];
      }
    });

    this.onNormalize(
      minMaxValues,
      'target',
      !this.state.isMinMaxValuesByTarget
    );
  }

  @boundMethod
  onNormalize(minMaxValues, minMaxSource, isMinMaxValuesByTarget = false) {
    this.setState({ minMaxValues, minMaxSource, isMinMaxValuesByTarget });
  }

  onPage(data) {
    const pageProp = `page${data[0].toUpperCase()}${data.slice(1)}`;
    return pageNum => {
      if (pageNum === this.state[pageProp] + 1) {
        this.onPageNext(data);
      } else if (pageNum === this.state[pageProp] - 1) {
        this.onPagePrev(data);
      } else {
        this.setState({ [pageProp]: Math.max(0, pageNum) });
      }
    };
  }

  async onPageNext(data) {
    const pageProp = `page${data[0].toUpperCase()}${data.slice(1)}`;
    const loadingProp = `isLoadingMore${data[0].toUpperCase()}${data.slice(1)}`;

    if (this.state[loadingProp]) return;

    const currentPage = this.state[pageProp];
    const currentNumSeeds = Object.keys(this.state.seeds).length;

    const numItems = this.state[data].length;

    if (data === 'seeds') {
      if (currentNumSeeds / (PER_PAGE_ITEMS * (currentPage + 1)) >= 1) {
        // Go to the next page as usual
        this.setState({ [pageProp]: currentPage + 1, [loadingProp]: false });
      } else {
        this.onLoadMoreSeeds();
      }
    } else if (numItems / (PER_PAGE_ITEMS * (currentPage + 1)) >= 1) {
      this.setState({ [pageProp]: currentPage + 1, [loadingProp]: false });
    }
  }

  onPagePrev(data) {
    const pageProp = `page${data[0].toUpperCase()}${data.slice(1)}`;
    const currentPage = this.state[pageProp];
    this.setState({ [pageProp]: Math.max(0, currentPage - 1) });
  }

  @boundMethod
  async onLoadMoreSeeds() {
    // Activate loading state
    await this.setState({ isLoadingMoreSeeds: true });

    // Re-train classifier and get new seeds once the training is done
    await this.onTrainingStart(this.onTrainingCheckSeeds);

    this.setState({ isLoadingMoreSeeds: false });
  }

  @boundMethod
  async onProgressStart(checker = this.onProgressCheck) {
    await api.getProgress(this.state.searchInfo.id);

    this.setState({
      isComputingProgress: true,
      progressCheckTimerId: setInterval(checker, PROGRESS_CHECK_INTERVAL)
    });
  }

  @boundMethod
  async onProgressCheck() {
    let progressInfo = await api.getProgress(this.state.searchInfo.id);

    const isErrorProgress =
      progressInfo.status !== 200
        ? "Could't get information on the progress computation."
        : false;
    progressInfo = isErrorProgress ? {} : progressInfo.body;

    if (progressInfo.isComputing) return;

    clearInterval(this.state.progressCheckTimerId);

    // Update state
    await this.setState({
      isErrorProgress,
      isComputingProgress: false,
      isLoadingProgress: false,
      progressCheckTimerId: null
    });

    if (isErrorProgress) return;

    // Update the classifier counter
    this.setState({
      progress: {
        unpredictabilityAll: progressInfo.unpredictabilityAll,
        unpredictabilityLabels: progressInfo.unpredictabilityLabels,
        predictionProbaChangeAll: progressInfo.predictionProbaChangeAll,
        predictionProbaChangeLabels: progressInfo.predictionProbaChangeLabels,
        convergenceAll: progressInfo.convergenceAll,
        convergenceLabels: progressInfo.convergenceLabels,
        divergenceAll: progressInfo.divergenceAll,
        divergenceLabels: progressInfo.divergenceLabels,
        numLabels: progressInfo.numLabels
      }
    });
  }

  @boundMethod
  async onTrainingStart(checker = this.onTrainingCheck) {
    const trainingInfo = await api.newClassifier(this.id);

    // Most likely the labels haven't changed
    if (trainingInfo.status === 409) {
      showInfo(this.props.pubSub, trainingInfo.body.error);
      return;
    }

    this.setState({
      isTraining: true,
      trainingCheckTimerId: setInterval(checker, TRAINING_CHECK_INTERVAL)
    });
  }

  @boundMethod
  async onTrainingCheck() {
    let classifierInfo = await api.getClassifier(this.state.searchInfo.id);

    const isErrorResults =
      classifierInfo.status !== 200
        ? "Could't get information on the training."
        : false;
    classifierInfo = isErrorResults ? {} : classifierInfo.body;

    if (classifierInfo.isTraining) return;

    clearInterval(this.state.trainingCheckTimerId);

    // Update state
    await this.setState({
      isErrorResults,
      isTraining: classifierInfo.isTraining,
      trainingCheckTimerId: null,
      pageResults: 0
    });

    if (isErrorResults) return;

    // Update the classifier counter
    await this.setState({
      searchInfo: update(this.state.searchInfo, {
        classifiers: { $set: this.state.searchInfo.classifiers + 1 }
      })
    });

    // Update progress
    this.loadProgress();

    // Get results first
    await this.loadResults();

    // And switch to the results tab
    this.props.setTab(TAB_RESULTS);
  }

  @boundMethod
  async onTrainingCheckSeeds() {
    let classifierInfo = await api.getClassifier(this.state.searchInfo.id);

    const isErrorSeeds =
      classifierInfo.status !== 200
        ? "Could't get information on the training."
        : false;
    classifierInfo = isErrorSeeds ? {} : classifierInfo.body;

    if (classifierInfo.isTraining) return;

    clearInterval(this.state.trainingCheckTimerId);

    // Update state
    await this.setState({
      isErrorSeeds,
      isTraining: classifierInfo.isTraining,
      trainingCheckTimerId: null,
      pageSeeds: 0
    });

    if (isErrorSeeds) return;

    // Update the classifier counter
    await this.setState({
      searchInfo: update(this.state.searchInfo, {
        classifiers: { $set: this.state.searchInfo.classifiers + 1 }
      })
    });

    // Update progress
    this.loadProgress();

    // Get new seeds
    this.loadSeeds();
  }

  onAction(action) {
    return value => {
      this.props[action](value);
    };
  }

  onChangeState(key) {
    return value => {
      this.setState({ [key]: value });
    };
  }

  getHgViewId(
    showAes = this.props.showAutoencodings,
    showProbs = this.isTrained,
    showSelection = !this.state.searchInfosAll
  ) {
    return searchId =>
      `${searchId}..${showAes ? 'e' : ''}${showProbs ? 'p' : ''}${
        showSelection ? 's' : ''
      }`;
  }

  /* -------------------------------- Render -------------------------------- */

  render() {
    if (this.state.isLoading || this.state.isInit) return this.renderLoader();
    if (this.state.isError) return this.renderError();

    if (this.state.searchInfo && this.state.searchInfo.status === 404)
      return this.renderNotFound();

    if (this.state.searchInfosAll) return this.renderListAllSearches();

    return this.renderSearch();
  }

  renderLoader() {
    return (
      <ContentWrapper name="search" isFullDimOnly={true}>
        <Content name="search" rel={true}>
          <SpinnerCenter />
        </Content>
      </ContentWrapper>
    );
  }

  renderError() {
    return (
      <ContentWrapper name="search" isFullDimOnly={true}>
        <Content name="search" rel={true}>
          <ErrorMsgCenter msg={this.state.isError} />
        </Content>
      </ContentWrapper>
    );
  }

  renderNotFound() {
    return (
      <NotFound
        title="O Peaks, Where Art Thou?"
        message={
          `No search with id ${this.id} was found. How about starting a new ` +
          'search?'
        }
      />
    );
  }

  renderListAllSearches() {
    const hgViewId = this.getHgViewId(false, false, false);

    return (
      <ContentWrapper name="search" isFullDimOnly={true}>
        <Content
          name="search"
          rel={true}
          hasSmallerTopBar={true}
          hasSubTopBar={true}
          bottomMargin={false}
          rightBarShow={this.props.rightBarShow}
          rightBarWidth={this.props.rightBarWidth}
        >
          <SearchSubTopBarAll
            viewportChanged={false}
            resetViewport={this.resetAllViewports}
          />
          {this.state.searchInfosAll.length ? (
            <ol className="no-list-style higlass-list">
              {this.state.searchInfosAll.map(info => (
                <li key={info.id} className="rel">
                  <div className="flex-c flex-jc-sb searches-metadata">
                    <Link to={`/search/${info.id}`} className="searches-name">
                      {info.name || `Search #${info.id}`}
                    </Link>
                    <div className="rel">
                      <ToolTip
                        align="center"
                        delayIn={2000}
                        delayOut={500}
                        title={
                          <span className="flex-c">
                            <span>Last updated</span>
                          </span>
                        }
                      >
                        <time dateTime={info.updated}>
                          {readableDate(info.updated, true)}
                        </time>
                      </ToolTip>
                    </div>
                  </div>
                  <div className="flex-c flex-jc-sb searches-progress">
                    <div className="flex-g-1">
                      Classifications: {info.classifications || 0}
                    </div>
                    <div className="flex-g-1">
                      Trainings: {info.trainings || 0}
                    </div>
                    <div className="flex-g-1">Hits: {info.hits || 0}</div>
                  </div>
                  <HiGlassViewer
                    height={info.viewHeight}
                    isGlobalMousePosition
                    isStatic
                    disableTrackMenu
                    viewConfigId={hgViewId(info.id)}
                  />
                  <Link to={`/search/${info.id}`} className="searches-continue">
                    Continue
                  </Link>
                </li>
              ))}
            </ol>
          ) : (
            <em>
              No searches found. How about starting a{' '}
              <Link to="/">new search</Link>?
            </em>
          )}
        </Content>
      </ContentWrapper>
    );
  }

  renderSearch() {
    const hgViewId = this.getHgViewId();

    return (
      <ContentWrapper name="search" isFullDimOnly={true}>
        <Content
          name="search"
          rel={true}
          hasRightBar={true}
          hasSubTopBar={true}
          isVertFlex={true}
          hasSmallerTopBar={true}
          bottomMargin={false}
          rightBarShow={this.props.rightBarShow}
          rightBarWidth={this.props.rightBarWidth}
        >
          <SearchSubTopBar
            isMinMaxValuesByTarget={this.state.isMinMaxValuesByTarget}
            normalize={this.normalizeByTarget}
            resetViewport={this.resetViewport}
            viewportChanged={this.state.viewportChanged}
          />
          <div className="rel search-target">
            <HiGlassViewer
              api={this.checkHgApi}
              height={
                this.isTrained
                  ? this.state.searchInfo.viewHeightMax
                  : this.state.searchInfo.viewHeightTarget
              }
              isGlobalMousePosition
              isStatic
              disableTrackMenu
              pixelPrecise
              viewConfigId={hgViewId(this.state.searchInfo.id)}
              viewConfigAdjustments={this.viewConfigAdjustments}
            />
          </div>
          <div className="rel flex-g-1 search-results">
            <SearchSubTopBarTabs
              minClassifications={this.state.info.minClassifications}
              numClassifications={this.state.searchInfo.classifications}
            />
            <div className="search-tabs">
              <TabContent
                className="full-dim flex-c flex-v"
                for={TAB_SEEDS}
                tabOpen={this.props.tab}
              >
                <SearchSeeds
                  classificationChangeHandler={this.classificationChangeHandler}
                  dataTracks={this.state.dataTracks}
                  info={this.state.info}
                  isError={this.state.isErrorSeeds}
                  isLoading={this.state.isLoadingSeeds}
                  isReady={this.isSeeded}
                  isTraining={this.state.isTraining}
                  itemsPerPage={PER_PAGE_ITEMS}
                  normalizationSource={this.state.minMaxSource}
                  normalizeBy={this.state.minMaxValues}
                  onNormalize={this.onNormalize}
                  onLoadMore={this.onLoadMoreSeeds}
                  onPage={this.onPageSeeds}
                  onTrainingStart={this.onTrainingStart}
                  onTrainingCheck={this.onTrainingCheck}
                  page={this.state.pageSeeds}
                  pageTotal={this.state.pageSeedsTotal}
                  results={this.state.seeds}
                  searchInfo={this.state.searchInfo}
                  windows={this.state.windows}
                />
              </TabContent>
              <TabContent
                className="full-dim flex-c flex-v"
                for={TAB_RESULTS}
                tabOpen={this.props.tab}
              >
                <SearchResults
                  classificationChangeHandler={this.classificationChangeHandler}
                  dataTracks={this.state.dataTracks}
                  info={this.state.info}
                  isError={this.state.isErrorResults}
                  isLoading={this.state.isLoadingResults}
                  isReady={this.isSeeded}
                  isTrained={this.isTrained}
                  isTraining={this.state.isTraining}
                  itemsPerPage={PER_PAGE_ITEMS}
                  loadResultsAgain={this.loadResults}
                  normalizationSource={this.state.minMaxSource}
                  normalizeBy={this.state.minMaxValues}
                  onNormalize={this.onNormalize}
                  onPage={this.onPageResults}
                  onChangePreditionProbBorder={this.onChangePreditionProbBorder}
                  onTrainingStart={this.onTrainingStart}
                  onTrainingCheck={this.onTrainingCheck}
                  page={this.state.pageResults}
                  predictionProbBorder={this.state.predictionProbBorder}
                  results={this.state.results}
                  resultsConflictsFp={this.state.resultsConflictsFp}
                  resultsConflictsFn={this.state.resultsConflictsFn}
                  resultsPredictionHistogram={
                    this.state.resultsPredictionHistogram
                  }
                  resultsPredictionProbBorder={
                    this.state.resultsPredictionProbBorder
                  }
                  searchInfo={this.state.searchInfo}
                  train={this.onTrainingBnd}
                  windows={this.state.windows}
                />
              </TabContent>
              <TabContent
                className="full-dim flex-c flex-v"
                for={TAB_CLASSIFICATIONS}
                tabOpen={this.props.tab}
              >
                <SearchClassifications
                  classificationChangeHandler={this.classificationChangeHandler}
                  dataTracks={this.state.dataTracks}
                  info={this.state.info}
                  isError={this.state.isErrorClassifications}
                  isLoading={this.state.isLoadingClassifications}
                  isTraining={this.state.isTraining}
                  itemsPerPage={PER_PAGE_ITEMS}
                  normalizationSource={this.state.minMaxSource}
                  normalizeBy={this.state.minMaxValues}
                  onNormalize={this.onNormalize}
                  onPage={this.onPageClassifications}
                  onTrainingStart={this.onTrainingStart}
                  onTrainingCheck={this.onTrainingCheck}
                  page={this.state.pageClassifications}
                  results={this.state.classifications}
                  searchInfo={this.state.searchInfo}
                  windows={this.state.windows}
                />
              </TabContent>
              <TabContent
                className="full-dim flex-c flex-v"
                for={TAB_SELECTION}
                tabOpen={this.props.tab}
              >
                <SearchSelection
                  classificationChangeHandler={this.classificationChangeHandler}
                  dataTracks={this.state.dataTracks}
                  info={this.state.info}
                  isError={this.state.isErrorSelection}
                  isLoading={this.state.isLoadingSelection}
                  isTraining={this.state.isTraining}
                  itemsPerPage={PER_PAGE_ITEMS}
                  normalizationSource={this.state.minMaxSource}
                  normalizeBy={this.state.minMaxValues}
                  onNormalize={this.onNormalize}
                  onPage={this.onPageSelection}
                  onTrainingStart={this.onTrainingStart}
                  onTrainingCheck={this.onTrainingCheck}
                  page={this.state.pageSelection}
                  results={this.state.selection}
                  searchInfo={this.state.searchInfo}
                  windows={this.state.windows}
                />
              </TabContent>
            </div>
          </div>
        </Content>
        <SearchRightBar
          isComputingProgress={this.state.isComputingProgress}
          isErrorProgress={this.state.isErrorProgress}
          progress={this.state.progress}
          pubSub={this.props.pubSub}
          searchInfo={this.state.searchInfo}
          widthSetterFinal={resizeTrigger}
        />
      </ContentWrapper>
    );
  }
}

Search.defaultProps = {
  id: -1,
  viewConfigId: 'default'
};

Search.propTypes = {
  match: PropTypes.object,
  pubSub: PropTypes.object.isRequired,
  rightBarShow: PropTypes.bool,
  rightBarTab: PropTypes.oneOfType([PropTypes.string, PropTypes.symbol])
    .isRequired,
  rightBarWidth: PropTypes.number,
  selectedRegions: PropTypes.array.isRequired,
  setHover: PropTypes.func,
  setRightBarShow: PropTypes.func,
  setRightBarTab: PropTypes.func,
  setTab: PropTypes.func.isRequired,
  showAutoencodings: PropTypes.bool,
  tab: PropTypes.oneOfType([PropTypes.string, PropTypes.symbol]).isRequired,
  viewConfig: PropTypes.object
};

const mapStateToProps = state => ({
  rightBarShow: state.present.searchRightBarShow,
  rightBarTab: state.present.searchRightBarTab,
  rightBarWidth: state.present.searchRightBarWidth,
  selectedRegions: state.present.searchSelection,
  showAutoencodings: state.present.showAutoencodings,
  tab: state.present.searchTab,
  viewConfig: state.present.viewConfig
});

const mapDispatchToProps = dispatch => ({
  setHover: windowId => dispatch(setSearchHover(windowId)),
  setRightBarShow: rightBarShow =>
    dispatch(setSearchRightBarShow(rightBarShow)),
  setRightBarTab: searchRightBarTab =>
    dispatch(setSearchRightBarTab(searchRightBarTab)),
  setTab: searchTab => dispatch(setSearchTab(searchTab))
});

export default withRouter(
  connect(mapStateToProps, mapDispatchToProps)(withPubSub(Search))
);
