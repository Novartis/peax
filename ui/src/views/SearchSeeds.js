import PropTypes from 'prop-types';
import React from 'react';

// Components
import Button from '../components/Button';
import HiglassResultListSingleHiGlass from '../components/HiglassResultListSingleHiGlass';
import SubTopBar from '../components/SubTopBar';
import SubTopBottomBarButtons from '../components/SubTopBottomBarButtons';

// Utils
import { Logger } from '../utils';

const logger = Logger('SearchSeeds'); // eslint-disable-line

const isTraining = onTrainingCheck => (
  <span>
    {'The classifier is training hard! '}
    <Button onClick={onTrainingCheck}>Check Status</Button>
  </span>
);

class SearchSeeds extends React.Component {
  constructor(props) {
    super(props);
    this.getResultsWrapperBound = this.getResultsWrapper.bind(this);
  }

  componentDidUpdate(prevProps) {
    if (this.props.page !== prevProps.page) this.resultsWrapper.scrollTop = 0;
  }

  get isFulfilled() {
    return (
      (this.props.searchInfo.classifications || 0) >
      this.props.info.minClassifications
    );
  }

  getResultsWrapper(ref) {
    this.resultsWrapper = ref;
  }

  render() {
    return (
      <div className="full-dim search-tab-wrapper">
        <SubTopBar>
          <SubTopBottomBarButtons className="flex-c flex-a-c no-list-style">
            {this.props.searchInfo.classifications >=
              this.props.searchInfo.config.min_classifications && (
              <li>
                Labeled {this.props.searchInfo.classifications} regions so far.
              </li>
            )}
            {this.props.searchInfo.classifications > 0 &&
              this.props.searchInfo.classifications <
                this.props.searchInfo.config.min_classifications && (
                <li>
                  Label another{' '}
                  {this.props.searchInfo.config.min_classifications -
                    this.props.searchInfo.classifications}{' '}
                  ({this.props.searchInfo.classifications} already labeled)
                  regions to train a classifier.
                </li>
              )}
            {this.props.searchInfo.classifications === 0 && (
              <li>
                Start labeling at least{' '}
                {this.props.searchInfo.config.min_classifications} regions to
                train a classifier.
              </li>
            )}
          </SubTopBottomBarButtons>
          <SubTopBottomBarButtons className="flex-c flex-a-c flex-jc-e no-list-style">
            <li>
              <Button
                isDisabled={
                  !this.props.isReady || this.props.isTraining === true
                }
                onClick={() => this.props.onTrainingStart()}
              >
                Train
              </Button>
            </li>
          </SubTopBottomBarButtons>
        </SubTopBar>
        <div
          ref={this.getResultsWrapperBound}
          className="search-tab-content search-results"
        >
          <HiglassResultListSingleHiGlass
            isError={this.props.isError}
            isLoading={this.props.isLoading}
            isEmpty={'No samples found!'}
            isMoreLoadable={true}
            isTraining={this.props.isTraining === true}
            isTrainingNodes={isTraining(this.props.onTrainingCheck)}
            itemsPerPage={this.props.itemsPerPage}
            list={Object.keys(this.props.results).map(windowId => ({
              classification: undefined,
              classificationChangeHandler: this.props
                .classificationChangeHandler,
              dataTracks: this.props.dataTracks,
              hidePredProb: true,
              normalizationSource: this.props.normalizationSource,
              normalizeBy: this.props.normalizeBy,
              onNormalize: this.props.onNormalize,
              searchId: this.props.searchInfo.id,
              viewHeight: this.props.searchInfo.viewHeight,
              windowId: parseInt(windowId, 10),
              windows: this.props.windows
            }))}
            page={this.props.page}
            pageTotal={Math.ceil(
              Object.keys(this.props.results).length / this.props.itemsPerPage
            )}
            onLoadMore={this.props.onLoadMore}
            onPage={this.props.onPage}
            textLoadMore="Train & Load More"
          />
        </div>
      </div>
    );
  }
}

SearchSeeds.defaultProps = {
  isLoading: true,
  isReady: null,
  isError: false,
  results: {},
  windows: {}
};

SearchSeeds.propTypes = {
  classificationChangeHandler: PropTypes.func.isRequired,
  dataTracks: PropTypes.array,
  info: PropTypes.object.isRequired,
  isError: PropTypes.oneOfType([PropTypes.bool, PropTypes.string]),
  isLoading: PropTypes.bool,
  isReady: PropTypes.bool,
  isTraining: PropTypes.bool,
  itemsPerPage: PropTypes.number,
  normalizationSource: PropTypes.oneOfType([
    PropTypes.number,
    PropTypes.string
  ]),
  normalizeBy: PropTypes.object,
  onNormalize: PropTypes.func.isRequired,
  onLoadMore: PropTypes.func.isRequired,
  onPage: PropTypes.func.isRequired,
  onTrainingCheck: PropTypes.func.isRequired,
  onTrainingStart: PropTypes.func.isRequired,
  page: PropTypes.number,
  pageTotal: PropTypes.number,
  results: PropTypes.object,
  searchInfo: PropTypes.object.isRequired,
  windows: PropTypes.object
};

export default SearchSeeds;
