import PropTypes from 'prop-types';
import React from 'react';

// Components
import ButtonIcon from './ButtonIcon';

// Styles
import './InfoBar.scss';

const InfoBar = props => (
  <header className={`info-bar ${props.isClose ? 'info-bar-is-close' : ''}`}>
    <div className={`info-bar-content ${props.wrap ? 'wrap' : ''}`}>
      {props.children}
    </div>
    {props.isClosable && (
      <div className="flex-c flex-a-c flex-jc-c rel info-bar-close">
        <ButtonIcon
          icon="arrow-bottom"
          iconMirrorH={!props.isClose}
          iconOnly={true}
          onClick={props.onClose}
        />
      </div>
    )}
  </header>
);

InfoBar.propTypes = {
  children: PropTypes.node.isRequired,
  isClose: PropTypes.bool,
  isClosable: PropTypes.bool,
  onClose: PropTypes.func,
  wrap: PropTypes.bool
};

export default InfoBar;
