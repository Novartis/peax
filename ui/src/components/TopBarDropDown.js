import PropTypes from 'prop-types';
import React from 'react';

// Components
import DropDown from './DropDown';

// Styles
import './TopBarDropDown.scss';

const className = classNames => `top-bar-drop-down ${classNames}`;

const TopBarDropDowUser = props => (
  <DropDown
    alignRight={props.alignRight}
    alignTop={props.alignTop}
    className={className(props.className)}
    closeOnOuterClick={props.closeOnOuterClick}
    id='TopBarDropDown'
  >
    {props.children}
  </DropDown>
);

TopBarDropDowUser.propTypes = {
  children: PropTypes.node.isRequired,
  className: PropTypes.string,
  closeOnOuterClick: PropTypes.bool,
  alignRight: PropTypes.bool,
  alignTop: PropTypes.bool,
};

export default TopBarDropDowUser;
