import PropTypes from 'prop-types';
import React from 'react';

// Components
import Message from './Message';

import './MessageCenter.scss';

const MessageCenter = props => (
  <div className="full-dim flex-c flex-a-c flex-jc-c message-center">
    <Message msg={props.msg} type={props.type}>
      {props.children}
    </Message>
  </div>
);

MessageCenter.propTypes = {
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

export default MessageCenter;
