import PropTypes from 'prop-types';
import React from 'react';

const TabTrigger = props => (
  <div className={`tab-trigger ${props.className}`}>
    {React.cloneElement(props.children, {
      isActive: props.tabOpen === props.for,
      onClick: () => props.tabChange(props.for)
    })}
  </div>
);

TabTrigger.defaultProps = {
  className: ''
};

TabTrigger.propTypes = {
  children: PropTypes.node.isRequired,
  className: PropTypes.string,
  for: PropTypes.oneOfType([PropTypes.string, PropTypes.symbol]).isRequired,
  tabChange: PropTypes.func.isRequired,
  tabOpen: PropTypes.oneOfType([PropTypes.string, PropTypes.symbol]).isRequired
};

export default TabTrigger;
