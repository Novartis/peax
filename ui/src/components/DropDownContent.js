import PropTypes from 'prop-types';
import React from 'react';

// Styles
import './DropDownContent.scss';

const DropDownContent = props => (
  <div className='flex-c flex-v drop-down-content'>
    {props.children}
  </div>
);

DropDownContent.propTypes = {
  children: PropTypes.node.isRequired,
};

export default DropDownContent;
