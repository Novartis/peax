import PropTypes from 'prop-types';
import React from 'react';
import { compose } from 'recompose';

// Components
import HiglassResult from './HiglassResult';
import MessageCenter from './MessageCenter';
import SpinnerCenter from './SpinnerCenter';

// HOCs
import withEither from './with-either';
import withList from './with-list';
import withMaybe from './with-maybe';
import withPagination from './with-pagination';

const getKey = props => props.windowId;
const isError = props => props.isError;
const isLoading = props => props.isLoading;
const isNull = props => !props.list;
const isEmpty = props => !props.list.length;
const isNotReady = props => props.isNotReady;
const isNotTrained = props => props.isNotTrained;
const isTraining = props => props.isTraining;

const ErrorMsg = props => (
  <MessageCenter msg={props.isError} type="error">
    {props.isErrorNodes}
  </MessageCenter>
);
ErrorMsg.propTypes = {
  isError: PropTypes.string,
  isErrorNodes: PropTypes.node
};

const IsEmptyMsg = props => (
  <MessageCenter msg={props.isEmptyText} type="warning">
    {props.isEmptyNodes}
  </MessageCenter>
);
IsEmptyMsg.propTypes = {
  isEmptyText: PropTypes.string,
  isEmptyNodes: PropTypes.node
};

const IsNotReadyMsg = props => (
  <MessageCenter msg={props.isNotReadyText} type="default">
    {props.isNotReadyNodes}
  </MessageCenter>
);
IsNotReadyMsg.propTypes = {
  isNotReadyText: PropTypes.string,
  isNotReadyNodes: PropTypes.node
};

const IsTrainingMsg = props => (
  <MessageCenter msg={props.isTrainingText} type="loading">
    {props.isTrainingNodes}
  </MessageCenter>
);
IsTrainingMsg.propTypes = {
  isTrainingText: PropTypes.string,
  isTrainingNodes: PropTypes.node
};

const IsNotTrainedMsg = props => (
  <MessageCenter msg={props.isNotTrainedText} type="info">
    {props.isNotTrainedNodes}
  </MessageCenter>
);
IsNotTrainedMsg.propTypes = {
  isNotTrainedText: PropTypes.string,
  isNotTrainedNodes: PropTypes.node
};

// Order of application is top to bottom
const HiglassResultsList = compose(
  withMaybe(isNull),
  withEither(isError, ErrorMsg),
  withEither(isLoading, SpinnerCenter),
  withEither(isNotReady, IsNotReadyMsg),
  withEither(isTraining, IsTrainingMsg),
  withEither(isNotTrained, IsNotTrainedMsg),
  withEither(isEmpty, IsEmptyMsg),
  withPagination(),
  withList(getKey)
)(HiglassResult);

export default HiglassResultsList;
