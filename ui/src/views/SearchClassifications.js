import PropTypes from 'prop-types';
import React from 'react';

// Components
import Button from '../components/Button';
import ButtonRadio from '../components/ButtonRadio';
import HiglassResultListSingleHiGlass from '../components/HiglassResultListSingleHiGlass';
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
import { Logger, numToClassif } from '../utils';

const logger = Logger('SearchClassifications'); // eslint-disable-line

const isEmpty = (
  <span>
    {'Nothing labeled! '}
    <Button onClick={() => setSearchTab(TAB_SEEDS)}>Get started</Button>
  </span>
);

const isTraining = onTrainingCheck => (
  <span>
    {'The classifier is training hard! '}
    <Button onClick={onTrainingCheck}>Check Status</Button>
  </span>
);

class SearchClassifications extends React.Component {
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

  render() {
    const sortOrder = this.state.sortOrder === 'desc' ? -1 : 1;

    const results = this.props.results
      .filter(
        win =>
          this.state.filterByClf === null ||
          numToClassif(win.classification) !== this.state.filterByClf
      )
      .sort((a, b) => {
        const aUpdated = Date.parse(a.updated);
        const bUpdated = Date.parse(b.updated);

        if (aUpdated < bUpdated) return -1 * sortOrder;
        if (aUpdated > bUpdated) return 1 * sortOrder;
        return 0;
      })
      .map(win => ({
        classification: numToClassif(win.classification),
        classificationChangeHandler: this.props.classificationChangeHandler,
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
            <li>Labeled {this.props.results.length} regions.</li>
          </SubTopBottomBarButtons>
          <SubTopBottomBarButtons className="flex-c flex-a-c flex-jc-e no-list-style">
            <li>
              <ButtonRadio
                label="Sort By Date"
                name="search-filter-by-classification"
                onClick={this.onSortOrderBnd}
                options={BUTTON_RADIO_SORT_ORDER_OPTIONS}
                selection={this.state.sortOrder}
              />
            </li>
            <li>
              <ToolTip
                align="center"
                delayIn={2000}
                delayOut={500}
                title={
                  <span className="flex-c">
                    <span>Exclude regions labeled positive or negative</span>
                  </span>
                }
              >
                <ButtonRadio
                  label="Exclude Labels"
                  name="search-filter-by-classification"
                  isDeselectable={true}
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

SearchClassifications.defaultProps = {
  isLoading: true,
  isError: false,
  results: [],
  windows: {}
};

SearchClassifications.propTypes = {
  classificationChangeHandler: PropTypes.func.isRequired,
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
  windows: PropTypes.object
};

export default SearchClassifications;
