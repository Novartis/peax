import PropTypes from 'prop-types';
import React from 'react';

// Styles
import './SubTopBar.scss';

const SubTopBar = props => (
  <header className={`sub-top-bar ${props.className}`}>
    <div
      className={`flex-c ${
        props.stretch ? 'flex-a-s' : 'flex-a-c'
      } flex-jc-sb sub-top-bar-content-wrap ${props.wrap ? 'wrap' : ''}`}
    >
      {props.children}
    </div>
  </header>
);

SubTopBar.defaultProps = {
  className: '',
  stretch: false,
  wrap: false
};

SubTopBar.propTypes = {
  children: PropTypes.node.isRequired,
  className: PropTypes.string,
  stretch: PropTypes.bool,
  wrap: PropTypes.bool
};

export default SubTopBar;
