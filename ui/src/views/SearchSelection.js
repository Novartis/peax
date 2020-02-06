import { boundMethod } from 'autobind-decorator';
import PropTypes from 'prop-types';
import React from 'react';
import { connect } from 'react-redux';

// Components
import Button from '../components/Button';
import ButtonIcon from '../components/ButtonIcon';
import ButtonRadio from '../components/ButtonRadio';
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
  TAB_RESULTS
} from '../configs/search';

// Utils
import { Logger, numToClassif } from '../utils';

const logger = Logger('SearchSelection'); // eslint-disable-line

const isEmpty = <span>{'Nothing selected!'}</span>;

const isTraining = onTrainingCheck => (
  <span>
    {'The classifier is training hard! '}
    <Button onClick={onTrainingCheck}>Check Status</Button>
  </span>
);

class SearchSelection extends React.Component {
  constructor(props) {
    super(props);

    this.getResultsWrapperBound = this.getResultsWrapper.bind(this);
    this.onFilterByClfBnd = this.onFilterByClf.bind(this);
    this.onSortOrderBnd = this.onSortOrder.bind(this);

    this.state = {
      filterByClf: null,
      sortOrder: 'desc'
    };
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
  async clearSelectionAndGoToResults() {
    this.props.clearSelection();
    this.props.setTab(TAB_RESULTS);
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
        const aProbability = a.probability || 0;
        const bProbability = b.probability || 0;

        if (aProbability < bProbability) return -1 * sortOrder;
        if (aProbability > bProbability) return 1 * sortOrder;
        return 0;
      })
      .map(win => ({
        classification: numToClassif(win.classification),
        classificationChangeHandler: this.props.classificationChangeHandler,
        classificationProb: win.probability,
        conflict: win.conflict,
        dataTracks: this.props.dataTracks,
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
            <li>
              Selected <strong>{this.props.results.length}</strong> regions.
            </li>
            <li>
              <ButtonIcon
                icon="cross"
                iconSmaller
                onClick={this.clearSelectionAndGoToResults}
              >
                Clear
              </ButtonIcon>
            </li>
          </SubTopBottomBarButtons>
          <SubTopBottomBarButtons className="flex-c flex-a-c flex-jc-e no-list-style">
            <li>
              <ToolTip
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
                  name="search-sort-by-genome-position"
                  onClick={this.onSortOrderBnd}
                  options={BUTTON_RADIO_SORT_ORDER_OPTIONS}
                  selection={this.state.sortOrder}
                />
              </ToolTip>
            </li>
            <li>
              <ToolTip
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
                isDisabled={this.props.isTraining === true}
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

SearchSelection.defaultProps = {
  isLoading: true,
  isError: false,
  results: [],
  windows: {}
};

SearchSelection.propTypes = {
  classificationChangeHandler: PropTypes.func.isRequired,
  clearSelection: PropTypes.func.isRequired,
  dataTracks: PropTypes.array,
  info: PropTypes.object.isRequired,
  isError: PropTypes.bool,
  isLoading: PropTypes.bool,
  isTraining: PropTypes.bool,
  itemsPerPage: PropTypes.number,
  normalizationSource: PropTypes.oneOfType([
    PropTypes.number,
    PropTypes.string
  ]),
  normalizeBy: PropTypes.object,
  onNormalize: PropTypes.func.isRequired,
  onPage: PropTypes.func.isRequired,
  onTrainingCheck: PropTypes.func.isRequired,
  onTrainingStart: PropTypes.func.isRequired,
  page: PropTypes.number,
  results: PropTypes.array,
  searchInfo: PropTypes.object.isRequired,
  selectedRegions: PropTypes.array.isRequired,
  setTab: PropTypes.func.isRequired,
  windows: PropTypes.object
};

const mapStateToProps = state => ({
  selectedRegions: state.present.searchSelection
});

const mapDispatchToProps = dispatch => ({
  clearSelection: () => dispatch(setSearchSelection([])),
  setTab: tab => dispatch(setSearchTab(tab))
});

export default connect(mapStateToProps, mapDispatchToProps)(SearchSelection);
