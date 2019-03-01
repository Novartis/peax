import PropTypes from 'prop-types';
import React from 'react';
import { connect } from 'react-redux';

// Components
import Badge from '../components/Badge';
import BarChart from '../components/BarChart';
import Icon from '../components/Icon';
import TabEntry from '../components/TabEntry';

// Utils
import { readableDate } from '../utils';

// Actions
import {
  setSearchRightBarInfoHelp,
  setSearchRightBarInfoMetadata
} from '../actions';

// Styles
import './Search.scss';

const SearchRightBarInfo = props => (
  <div className="right-bar-info flex-c flex-v full-wh">
    <ul className="search-right-bar-padding no-list-style compact-list compact-list-with-padding">
      <li className="flex-c flex-v">
        <span className="label">Unpredictability</span>
        <BarChart
          x={props.progress.numLabels}
          y={props.progress.unpredictabilityAll}
          y2={props.progress.unpredictabilityLabels}
          parentWidth={props.rightBarWidth}
        />
      </li>
      <li className="flex-c flex-v">
        <span className="label">Unstability</span>
        {console.log(props.progress)}
        <BarChart
          x={props.progress.numLabels}
          y={props.progress.predictionProbaChangeAll}
          y2={props.progress.predictionProbaChangeLabels}
          parentWidth={props.rightBarWidth}
        />
      </li>
      <li className="flex-c flex-v">
        <span className="label">Convergence</span>
        <BarChart
          x={props.progress.numLabels}
          y={props.progress.convergenceAll}
          y2={props.progress.convergenceLabels}
          parentWidth={props.rightBarWidth}
        />
      </li>
      <li className="flex-c flex-jc-sb">
        <span className="label flex-g-1"># Labels</span>
        <Badge
          isBordered={true}
          value={props.searchInfo.classifications || 0}
        />
      </li>
      <li className="flex-c flex-jc-sb">
        <span className="label flex-g-1"># Trainings</span>
        <Badge
          isBordered={true}
          levelPoor={0}
          levelOkay={1}
          levelGood={3}
          value={props.searchInfo.classifiers || 0}
        />
      </li>
      <li className="flex-c flex-jc-sb">
        <span className="label flex-g-1"># Hits</span>
        <Badge isBordered={true} value={props.searchInfo.hits || 0} />
      </li>
    </ul>
    <TabEntry
      isOpen={props.searchRightBarInfoMetadata}
      title="Metadata"
      toggle={props.toggleSearchRightBarInfoMetadata}
    >
      <div className="search-right-bar-padding">
        <ul className="no-list-style compact-list">
          <li className="flex-c flex-jc-sb">
            <span className="label flex-g-1">Search ID</span>
            <span className="value">{props.searchInfo.id || '?'}</span>
          </li>
          <li className="flex-c flex-jc-sb">
            <span className="label flex-g-1">Last Update</span>
            <span className="value">
              {props.searchInfo.updated
                ? readableDate(props.searchInfo.updated)
                : '?'}
            </span>
          </li>
        </ul>
      </div>
      <div className="flex-c" />
    </TabEntry>
    <TabEntry
      isOpen={props.searchRightBarInfoHelp}
      title="Help"
      toggle={props.toggleSearchRightBarInfoHelp}
    >
      <div className="search-right-bar-padding">
        <ul className="no-list-style compact-list">
          <li>
            <strong>Hit: </strong>
            <span>
              A hit is a genomic window that is classified positive by the
              random forest model based on your labels and your probability
              threshold.
            </span>
          </li>
          <li>
            <strong>Label: </strong>
            <span>
              A label is the manual assignment of either a positive (
              <Icon iconId="checkmark" inline />) or negative (
              <Icon iconId="cross" inline />) class to a genomic window.
            </span>
          </li>
          <li>
            <strong>Training: </strong>
            <span>
              After labeling some genomic windows, a random forest classifier
              will be train.
            </span>
          </li>
          <li>
            <strong>Uncertainty: </strong>
            <span>
              Class prediction uncertainty is defined as the mean distance of
              classification probabilities from 0.5. An uncertainty of 1 is the
              worst and means that the classifier predicts the class for window
              with probability 0.5. An uncertainty of 0 is the best and means
              that the classifier predicts the class for window with either 1.0
              or 0.0.
            </span>
          </li>
          <li>
            <strong>Variance: </strong>
            <span>
              Class prediction variance is defined as the mean variance of class
              predictions based on confidence intervals as proposed by{' '}
              <a
                href="http://jmlr.org/papers/v15/wager14a.html"
                target="_blank"
                rel="noopener noreferrer"
              >
                Wager et al. 2014
              </a>
            </span>
          </li>
        </ul>
      </div>
      <div className="flex-c" />
    </TabEntry>
  </div>
);

SearchRightBarInfo.propTypes = {
  progress: PropTypes.object.isRequired,
  searchInfo: PropTypes.object.isRequired,
  rightBarWidth: PropTypes.number.isRequired,
  toggleSearchRightBarInfoHelp: PropTypes.func.isRequired,
  toggleSearchRightBarInfoMetadata: PropTypes.func.isRequired,
  searchRightBarInfoHelp: PropTypes.bool.isRequired,
  searchRightBarInfoMetadata: PropTypes.bool.isRequired
};

const mapStateToProps = state => ({
  rightBarWidth: state.present.searchRightBarWidth,
  searchRightBarInfoHelp: state.present.searchRightBarInfoHelp,
  searchRightBarInfoMetadata: state.present.searchRightBarInfoMetadata
});

const mapDispatchToProps = dispatch => ({
  toggleSearchRightBarInfoHelp: isOpen =>
    dispatch(setSearchRightBarInfoHelp(!isOpen)),
  toggleSearchRightBarInfoMetadata: isOpen =>
    dispatch(setSearchRightBarInfoMetadata(!isOpen))
});

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(SearchRightBarInfo);
