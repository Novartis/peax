import PropTypes from 'prop-types';
import React from 'react';

// Styles
import './RightBarSubTopBar.scss';

const classNames = () => {
  const className = 'flex-c right-bar-sub-top-bar';

  return className;
};


const RightBarSubTopBar = props => (
  <header className={classNames(props)}>
    {props.children}
  </header>
);

RightBarSubTopBar.propTypes = {
  children: PropTypes.node.isRequired,
};

export default RightBarSubTopBar;
