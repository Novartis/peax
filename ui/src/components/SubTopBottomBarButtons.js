import { PropTypes } from 'prop-types';
import React from 'react';

// Styles
import './SubTopBottomBarButtons.scss';

const classNames = (props) => {
  let className = 'sub-top-bottom-bar-buttons';

  className += ` ${props.className}`;
  className += props.iconOnly ? ' button-icon-only' : '';
  className += props.isActive ? ' is-active' : '';

  return className;
};

const SubTopBottomBarButtons = props => (
  <ul className={classNames(props)}>
    {props.children}
  </ul>
);

SubTopBottomBarButtons.propTypes = {
  children: PropTypes.node,
  className: PropTypes.string,
};

export default SubTopBottomBarButtons;
