import PropTypes from 'prop-types';
import React from 'react';
import { connect } from 'react-redux';

// Components
import Button from '../components/Button';
import ButtonIcon from '../components/ButtonIcon';
import ButtonRadio from '../components/ButtonRadio';
import DropDownSlider from '../components/DropDownSlider';
import HiglassResultList from '../components/HiglassResultList';
import SubTopBar from '../components/SubTopBar';
import SubTopBottomBarButtons from '../components/SubTopBottomBarButtons';
import ToolTip from '../components/ToolTip';

// Actions
import { setSearchTab } from '../actions';

// Configs
import {
  BUTTON_RADIO_FILTER_CLASSIFICATION_OPTIONS,
  BUTTON_RADIO_SORT_ORDER_OPTIONS,
  TAB_SEEDS
} from '../configs/search';

// Utils
import { Logger, numToCassif } from '../utils';

const logger = Logger('SearchResults'); // eslint-disable-line

const isNotReady = onGetStarted => (
  <span>
    {'More seeds need to be classified. '}
    <Button onClick={onGetStarted}>Get started</Button>
  </span>
);

const isNotTrained = onTrainingStart => (
  <span>
    {'Classifier needs to be trained first. '}
    <Button onClick={onTrainingStart}>Start training</Button>
  </span>
);

const isTraining = onTrainingCheck => (
  <span>
    {'The classifier is training hard! '}
    <Button onClick={onTrainingCheck}>Check Status</Button>
  </span>
);

const isEmpty = <span>{'Nothing found! This is bad.'}</span>;

class SearchResults extends React.Component {
  constructor(props) {
    super(props);
    this.getResultsWrapperBound = this.getResultsWrapper.bind(this);
    this.onFilterByClfBnd = this.onFilterByClf.bind(this);
    this.onSortOrderBnd = this.onSortOrder.bind(this);

    this.state = {
      filterByClf: ['positive', 'negative'],
      sortOrder: 'desc'
    };

    // Bound methods
    this.goToSeedsBound = this.goToSeeds.bind(this);
  }

  componentDidUpdate(prevProps) {
    if (this.props.page !== prevProps.page) this.resultsWrapper.scrollTop = 0;
  }

  getResultsWrapper(ref) {
    this.resultsWrapper = ref;
  }

  async onFilterByClf(clf, deselectedClf) {
    const filterByClf = [...this.state.filterByClf];

    if (clf) filterByClf.push(clf);
    if (deselectedClf) {
      const idx = filterByClf.indexOf(deselectedClf);
      if (idx >= 0) filterByClf.splice(idx, 1);
    }

    this.setState({ filterByClf });
    this.props.onPage(0);
  }

  async onSortOrder(order) {
    this.setState({ sortOrder: order });
    this.props.onPage(0);
  }

  async goToSeeds() {
    this.props.setTab(TAB_SEEDS);
  }

  render() {
    const sortOrder = this.state.sortOrder === 'desc' ? -1 : 1;
    const results = this.props.results
      .filter(
        win =>
          numToCassif(win.classification) === 'neutral' ||
          this.state.filterByClf.includes(numToCassif(win.classification))
      )
      .sort((a, b) => {
        const aProbability = a.probability;
        const bProbability = b.probability;

        if (aProbability < bProbability) return -1 * sortOrder;
        if (aProbability > bProbability) return 1 * sortOrder;
        return 0;
      })
      .map(win => ({
        classification: numToCassif(win.classification),
        classificationChangeHandler: this.props.classificationChangeHandler,
        classificationProb: win.probability,
        dataTracks: this.props.dataTracks,
        isInfoSideBar: true,
        normalizationSource: this.props.normalizationSource,
        normalizeBy: this.props.normalizeBy,
        onNormalize: this.props.onNormalize,
        searchId: this.props.searchInfo.id,
        viewHeight: this.props.searchInfo.viewHeight,
        windowId: win.windowId,
        windows: this.props.windows
      }));

    return (
      <div className="full-dim search-tab-wrapper">
        <SubTopBar>
          <SubTopBottomBarButtons className="flex-c flex-a-c no-list-style">
            {!this.props.isLoading &&
              !this.props.isError &&
              this.props.isTrained && (
                <li className="flex-c flex-a-c">
                  <span className="m-r-0-25">
                    Found <strong>{results.length}</strong> regions with a
                    prediction probability &ge;
                  </span>
                  <DropDownSlider
                    histogram={this.props.resultsPredictionHistogram}
                    histogramNorm={results.length}
                    onChange={this.props.onChangePreditionProbBorder}
                    reversed
                    value={this.props.predictionProbBorder}
                  />
                  {this.props.predictionProbBorder !==
                    this.props.resultsPredictionProbBorder && (
                    <ButtonIcon
                      className="m-l-0-25"
                      icon="reset"
                      iconOnly
                      isIconRotationOnFocus
                      onClick={this.props.loadResultsAgain}
                    />
                  )}
                </li>
              )}
          </SubTopBottomBarButtons>
          <SubTopBottomBarButtons className="flex-c flex-a-c flex-jc-e no-list-style">
            <li>
              <ToolTip
                align="center"
                delayIn={2000}
                delayOut={500}
                title={
                  <span className="flex-c">
                    <span>Download Results</span>
                  </span>
                }
              >
                <ButtonIcon
                  icon="download"
                  iconOnly={true}
                  isDisabled={true}
                  onClick={this.props.onTrainingStart}
                />
              </ToolTip>
            </li>
            <li className="flex-c flex-a-c">
              <span className="m-r-0-25">Sort</span>
              <ToolTip
                align="center"
                delayIn={2000}
                delayOut={500}
                title={
                  <span className="flex-c">
                    <span>Sort by prediction prob.</span>
                  </span>
                }
              >
                <ButtonRadio
                  name="search-filter-by-classification"
                  onClick={this.onSortOrderBnd}
                  options={BUTTON_RADIO_SORT_ORDER_OPTIONS}
                  selection={this.state.sortOrder}
                />
              </ToolTip>
            </li>
            <li className="flex-c flex-a-c">
              <span className="m-r-0-25">Filter</span>
              <ToolTip
                align="center"
                delayIn={2000}
                delayOut={500}
                title={
                  <span className="flex-c">
                    <span>Filter by label (incl.)</span>
                  </span>
                }
              >
                <ButtonRadio
                  name="search-filter-by-classification"
                  isDeselectable={true}
                  isMultiple={true}
                  onClick={this.onFilterByClfBnd}
                  options={BUTTON_RADIO_FILTER_CLASSIFICATION_OPTIONS}
                  selection={this.state.filterByClf}
                />
              </ToolTip>
            </li>
            <li>
              <Button
                isDisabled={this.props.isTraining === true}
                onClick={() => this.props.onTrainingStart()}
              >
                Re-Train
              </Button>
            </li>
          </SubTopBottomBarButtons>
        </SubTopBar>
        <div ref={this.getResultsWrapperBound} className="search-tab-content">
          <HiglassResultList
            isError={this.props.isError}
            isLoading={this.props.isLoading}
            isNotReady={this.props.isReady === false}
            isNotReadyNodes={isNotReady(this.goToSeedsBound)}
            isNotTrained={this.props.isTrained === false}
            isNotTrainedNodes={isNotTrained(this.props.onTrainingStart)}
            isTraining={this.props.isTraining === true}
            isTrainingNodes={isTraining(this.props.onTrainingCheck)}
            isEmptyNodes={isEmpty}
            itemsPerPage={this.props.itemsPerPage}
            list={results}
            page={this.props.page}
            pageTotal={Math.ceil(results.length / this.props.itemsPerPage)}
            onPage={this.props.onPage}
          />
        </div>
      </div>
    );
  }
}

SearchResults.defaultProps = {
  dataTracks: [],
  isError: false,
  isLoading: true,
  isReady: null,
  isTrained: false,
  isTraining: null,
  results: [],
  resultsPredictionHistogram: [],
  resultsPredictionProbBorder: null,
  searchInfo: {}
};

SearchResults.propTypes = {
  classificationChangeHandler: PropTypes.func.isRequired,
  dataTracks: PropTypes.array,
  info: PropTypes.object.isRequired,
  isError: PropTypes.oneOfType([PropTypes.bool, PropTypes.string]),
  isLoading: PropTypes.bool,
  isReady: PropTypes.bool,
  isTrained: PropTypes.bool,
  isTraining: PropTypes.bool,
  itemsPerPage: PropTypes.number,
  loadResultsAgain: PropTypes.func.isRequired,
  normalizationSource: PropTypes.oneOfType([
    PropTypes.number,
    PropTypes.string
  ]),
  normalizeBy: PropTypes.object,
  onNormalize: PropTypes.func.isRequired,
  onPage: PropTypes.func.isRequired,
  onChangePreditionProbBorder: PropTypes.func.isRequired,
  onTrainingStart: PropTypes.func.isRequired,
  onTrainingCheck: PropTypes.func.isRequired,
  page: PropTypes.number,
  pageTotal: PropTypes.number,
  predictionProbBorder: PropTypes.number.isRequired,
  results: PropTypes.array.isRequired,
  resultsPredictionHistogram: PropTypes.array,
  resultsPredictionProbBorder: PropTypes.number,
  searchInfo: PropTypes.object,
  setTab: PropTypes.func,
  windows: PropTypes.object
};

const mapStateToProps = (/* state */) => ({});

const mapDispatchToProps = dispatch => ({
  setTab: tab => dispatch(setSearchTab(tab))
});

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(SearchResults);
