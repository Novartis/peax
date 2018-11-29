import PropTypes from 'prop-types';
import React from 'react';

// Styles
import './TabContent.scss';

const classNames = props => {
  let className = 'tab-content';

  className += ` ${props.className}`;
  className += props.tabOpen === props.for ? ' is-open' : '';

  return className;
};

const TabContent = props => (
  <div className={classNames(props)}>
    {props.tabOpen === props.for && props.children}
  </div>
);

TabContent.propTypes = {
  children: PropTypes.node.isRequired,
  className: PropTypes.string,
  for: PropTypes.oneOfType([PropTypes.string, PropTypes.symbol]).isRequired,
  tabOpen: PropTypes.oneOfType([PropTypes.string, PropTypes.symbol]).isRequired
};

export default TabContent;
