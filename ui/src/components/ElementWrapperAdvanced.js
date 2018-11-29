import PropTypes from 'prop-types';
import React from 'react';
import { compose } from 'recompose';

// Components
import ElementWrapper from './ElementWrapper';
import MessageCenter from './MessageCenter';
import SpinnerCenter from './SpinnerCenter';

// HOCs
import withEither from './with-either';

const isNotFound = props => props.isNotFound;
const isError = props => props.isError;
const isLoading = props => props.isLoading;

const ErrorMsg = props => (
  <MessageCenter msg={props.isError} type="error">
    {props.isErrorNodes}
  </MessageCenter>
);
ErrorMsg.propTypes = {
  isError: PropTypes.string,
  isErrorNodes: PropTypes.node
};

const NotFoundMsg = props => (
  <MessageCenter msg={props.isNotFound} type="default">
    {props.isNotFoundNodes}
  </MessageCenter>
);
NotFoundMsg.propTypes = {
  isNotFound: PropTypes.string,
  isNotFoundNodes: PropTypes.node
};

const ElementWrapperAdvanced = compose(
  withEither(isError, ErrorMsg),
  withEither(isNotFound, NotFoundMsg),
  withEither(isLoading, SpinnerCenter)
)(ElementWrapper);

export default ElementWrapperAdvanced;
