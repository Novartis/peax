import PropTypes from 'prop-types';
import React from 'react';

// Styles
import './RightBarContent.scss';

const classNames = () => {
  const className = 'full-dim right-bar-content';

  return className;
};


const RightBarContent = props => (
  <div className={classNames(props)}>
    {props.children}
  </div>
);

RightBarContent.propTypes = {
  children: PropTypes.node.isRequired,
};

export default RightBarContent;
