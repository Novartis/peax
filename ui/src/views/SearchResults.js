import { boundMethod } from 'autobind-decorator';
import PropTypes from 'prop-types';
import React from 'react';
import { connect } from 'react-redux';

// Components
import Button from '../components/Button';
import ButtonIcon from '../components/ButtonIcon';
import ButtonRadio from '../components/ButtonRadio';
import DropDownMenu from '../components/DropDownMenu';
import DropDownSlider from '../components/DropDownSlider';
import HiglassResultListSingleHiGlass from '../components/HiglassResultListSingleHiGlass';
import SubTopBar from '../components/SubTopBar';
import SubTopBottomBarButtons from '../components/SubTopBottomBarButtons';
import ToolTip from '../components/ToolTip';

// Actions
import { setSearchSelection, setSearchTab } from '../actions';

// Configs
import {
  BUTTON_RADIO_CLASSIFICATION_OPTIONS,
  BUTTON_RADIO_SORT_ORDER_OPTIONS,
  TAB_SEEDS,
  TAB_SELECTION
} from '../configs/search';

// Utils
import { Logger, numToClassif } from '../utils';

const logger = Logger('SearchResults'); // eslint-disable-line

const isNotReady = onGetStarted => (
  <span>
    {'More samples need to be labeled. '}
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
      filterByClf: null,
      sortOrder: 'desc'
    };

    this.conflicts = [
      {
        name: 'False positives',
        description:
          'Windows labeled as uninteresting/negative that are predicted to be positive by the classifier',
        getNumber: () => this.props.resultsConflictsFp.length,
        onClick: () => this.onSelectConflicts('fp')
      },
      {
        name: 'False negatives',
        description:
          'Windows labeled as interesting/positive that are not predicted to be positive by the classifier',
        getNumber: () => this.props.resultsConflictsFn.length,
        onClick: () => this.onSelectConflicts('fb')
      }
    ];
  }

  componentDidUpdate(prevProps) {
    if (this.props.page !== prevProps.page) this.resultsWrapper.scrollTop = 0;
  }

  getResultsWrapper(ref) {
    this.resultsWrapper = ref;
  }

  async onFilterByClf(clf) {
    this.setState({ filterByClf: clf });
    this.props.onPage(0);
  }

  async onSortOrder(order) {
    this.setState({ sortOrder: order });
    this.props.onPage(0);
  }

  @boundMethod
  async goToSeeds() {
    this.props.setTab(TAB_SEEDS);
  }

  @boundMethod
  async onSelectConflicts(type) {
    const conflicts =
      type === 'fp'
        ? this.props.resultsConflictsFp
        : this.props.resultsConflictsFn;

    if (conflicts.length === 0) return;

    await this.props.setSelection(conflicts.map(conflict => conflict.windowId));

    this.props.setTab(TAB_SELECTION);
  }

  render() {
    const sortOrder = this.state.sortOrder === 'desc' ? -1 : 1;
    const results = this.props.results
      .filter(
        win =>
          this.state.filterByClf === null ||
          numToClassif(win.classification) !== this.state.filterByClf
      )
      .sort((a, b) => {
        const aProbability = a.probability;
        const bProbability = b.probability;

        if (aProbability < bProbability) return -1 * sortOrder;
        if (aProbability > bProbability) return 1 * sortOrder;
        return 0;
      })
      .map(win => ({
        classification: numToClassif(win.classification),
        classificationChangeHandler: this.props.classificationChangeHandler,
        classificationProb: win.probability,
        dataTracks: this.props.dataTracks,
        isInfoSideBar: true,
        normalizationSource: this.props.normalizationSource,
        normalizeBy: this.props.normalizeBy,
        onNormalize: this.props.onNormalize,
        searchId: this.props.searchInfo.id,
        valueScalesLocked: this.props.searchInfo.valueScalesLocked,
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
            {!!(
              this.props.resultsConflictsFp.length ||
              this.props.resultsConflictsFn.length
            ) && (
              <li>
                <DropDownMenu
                  className="warning"
                  items={this.conflicts}
                  trigger={`${this.props.resultsConflictsFp.length +
                    this.props.resultsConflictsFn.length} conflicts`}
                  triggerIcon="warning"
                />
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
            <li>
              <ToolTip
                label="Sort"
                align="center"
                delayIn={2000}
                delayOut={500}
                title={
                  <span className="flex-c">
                    <span>Sort by prediction probability</span>
                  </span>
                }
              >
                <ButtonRadio
                  label="Sort by pred. prob."
                  name="search-filter-by-classification"
                  onClick={this.onSortOrderBnd}
                  options={BUTTON_RADIO_SORT_ORDER_OPTIONS}
                  selection={this.state.sortOrder}
                />
              </ToolTip>
            </li>
            <li>
              <ToolTip
                label="Filter"
                align="center"
                delayIn={2000}
                delayOut={500}
                title={
                  <span className="flex-c">
                    <span>
                      Exclude regions labeled positive, neutral, or negative
                    </span>
                  </span>
                }
              >
                <ButtonRadio
                  label="Exclude Labels"
                  name="search-filter-by-classification"
                  isDeselectable={true}
                  onClick={this.onFilterByClfBnd}
                  options={BUTTON_RADIO_CLASSIFICATION_OPTIONS}
                  selection={this.state.filterByClf}
                />
              </ToolTip>
            </li>
            <li>
              <Button
                isBold={this.props.isTraining !== true}
                isDisabled={this.props.isTraining === true}
                isPrimary
                onClick={() => this.props.onTrainingStart()}
              >
                Re-Train
              </Button>
            </li>
          </SubTopBottomBarButtons>
        </SubTopBar>
        <div ref={this.getResultsWrapperBound} className="search-tab-content">
          <HiglassResultListSingleHiGlass
            isError={this.props.isError}
            isLoading={this.props.isLoading}
            isNotReady={this.props.isReady === false}
            isNotReadyNodes={isNotReady(this.goToSeeds)}
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
  predictionProbBorder: PropTypes.number.isRequired,
  results: PropTypes.array.isRequired,
  resultsConflictsFp: PropTypes.array.isRequired,
  resultsConflictsFn: PropTypes.array.isRequired,
  resultsPredictionHistogram: PropTypes.array,
  resultsPredictionProbBorder: PropTypes.number,
  searchInfo: PropTypes.object,
  setSelection: PropTypes.func.isRequired,
  setTab: PropTypes.func.isRequired,
  windows: PropTypes.object
};

const mapStateToProps = (/* state */) => ({});

const mapDispatchToProps = dispatch => ({
  setSelection: selection => dispatch(setSearchSelection(selection)),
  setTab: tab => dispatch(setSearchTab(tab))
});

export default connect(mapStateToProps, mapDispatchToProps)(SearchResults);
