import PropTypes from 'prop-types';
import React from 'react';

// Styles
import './BottomBar.scss';

const BottomBar = props => (
  <footer className='flex-c flex-jc-sb bottom-bar'>
    {props.children}
  </footer>
);

BottomBar.propTypes = {
  children: PropTypes.node.isRequired,
};

export default BottomBar;
