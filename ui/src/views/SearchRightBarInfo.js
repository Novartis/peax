import PropTypes from 'prop-types';
import React from 'react';
import { connect } from 'react-redux';

// Components
import Badge from '../components/Badge';
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
      <li className="flex-c flex-jc-sb">
        <span className="label flex-g-1">Classifications</span>
        <Badge
          isBordered={true}
          value={props.searchInfo.classifications || 0}
        />
      </li>
      <li className="flex-c flex-jc-sb">
        <span className="label flex-g-1">Classifications</span>
        <Badge
          isBordered={true}
          value={props.searchInfo.classifications || 0}
        />
      </li>
      <li className="flex-c flex-jc-sb">
        <span className="label flex-g-1">Trainings</span>
        <Badge
          isBordered={true}
          levelPoor={0}
          levelOkay={1}
          levelGood={3}
          value={props.searchInfo.classifiers || 0}
        />
      </li>
      <li className="flex-c flex-jc-sb">
        <span className="label flex-g-1">Hits</span>
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
            <strong>Hit: </strong>
            <span>
              A hit is a genomic window that is classified positive by the
              random forest model based on your labels and your probability
              threshold.
            </span>
          </li>
        </ul>
      </div>
      <div className="flex-c" />
    </TabEntry>
  </div>
);

SearchRightBarInfo.defaultProps = {
  searchInfo: {}
};

SearchRightBarInfo.propTypes = {
  searchInfo: PropTypes.object,
  toggleSearchRightBarInfoHelp: PropTypes.func,
  toggleSearchRightBarInfoMetadata: PropTypes.func,
  searchRightBarInfoLensLocation: PropTypes.bool,
  searchRightBarInfoLensValue: PropTypes.bool,
  searchRightBarInfoHelp: PropTypes.bool,
  searchRightBarInfoMetadata: PropTypes.bool
};

const mapStateToProps = state => ({
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
