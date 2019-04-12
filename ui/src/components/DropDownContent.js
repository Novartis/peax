import PropTypes from 'prop-types';
import React from 'react';

// Styles
import './DropDownContent.scss';

const DropDownContent = props => (
  <div className="drop-down-content">
    <div className="flex-c flex-v">{props.children}</div>
  </div>
);

DropDownContent.propTypes = {
  children: PropTypes.node.isRequired
};

export default DropDownContent;
