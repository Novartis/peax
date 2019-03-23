import PropTypes from 'prop-types';
import React from 'react';

// Components
import Icon from './Icon';
import Spinner from './Spinner';

import './Message.scss';

const getIcon = type => {
  switch (type) {
    case 'error':
    case 'warning':
      return 'warning';

    case 'help':
      return 'help';

    case 'info':
    default:
      return 'info-disc';
  }
};

const Message = props => (
  <div className={`flex-c flex-v flex-a-c message message-${props.type}`}>
    {props.type === 'loading' ? (
      <Spinner />
    ) : (
      <Icon iconId={getIcon(props.type)} />
    )}
    <p>{props.msg ? props.msg : props.children}</p>
    <div>{props.msg && props.children && props.children}</div>
  </div>
);

Message.defaultProps = {
  type: 'default'
};

Message.propTypes = {
  children: PropTypes.node,
  msg: PropTypes.string,
  type: PropTypes.oneOf([
    'default',
    'help',
    'info',
    'warning',
    'error',
    'loading'
  ])
};

export default Message;
