import { PropTypes } from 'prop-types';
import React from 'react';

// Styles
import './DropArea.scss';

const DropArea = props => (
  <div className={`flex-c flex-a-c rel drop-area ${props.className}`}>
    {props.children}
  </div>
);

DropArea.defaultProps = {
  className: '',
};

DropArea.propTypes = {
  children: PropTypes.node.isRequired,
  className: PropTypes.string,
};

export default DropArea;
