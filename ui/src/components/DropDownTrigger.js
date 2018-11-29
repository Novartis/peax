import PropTypes from 'prop-types';
import React from 'react';

const DropDownTrigger = props => (
  <div className="drop-down-trigger">
    {React.cloneElement(props.children, {
      isActive: props.dropDownIsOpen,
      onClick: props.dropDownToggle
    })}
  </div>
);

DropDownTrigger.propTypes = {
  children: PropTypes.node.isRequired,
  dropDownIsOpen: PropTypes.bool,
  dropDownToggle: PropTypes.func
};

export default DropDownTrigger;
