import PropTypes from 'prop-types';
import React from 'react';

// Components
import Icon from './Icon';

import './ErrorMsg.scss';

const ErrorMsg = props => (
  <div className='error-msg flex-c flex-v flex-a-c'>
    <Icon iconId='warning' />
    <p>{props.msg}</p>
  </div>
);

ErrorMsg.propTypes = {
  msg: PropTypes.string.isRequired,
};

export default ErrorMsg;
