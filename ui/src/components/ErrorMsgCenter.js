import PropTypes from 'prop-types';
import React from 'react';

// Components
import ErrorMsg from './ErrorMsg';

const ErrorMsgCenter = props => (
  <div className='full-dim flex-c flex-a-c flex-jc-c error-msg-center'>
    <ErrorMsg msg={props.msg} />
  </div>
);

ErrorMsgCenter.propTypes = {
  msg: PropTypes.string.isRequired,
};

export default ErrorMsgCenter;
