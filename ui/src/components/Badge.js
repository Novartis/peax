import { PropTypes } from 'prop-types';
import React from 'react';

// Styles
import './Badge.scss';

const classNames = props => {
  let className = 'badge';

  let level = 'poor';
  if (props.value >= props.levelOkay) level = 'okay';
  if (props.value >= props.levelGood) level = 'good';

  className += ` badge-level-${level}`;

  if (props.isBordered) className += ' badge-border';

  return className;
};

const Badge = props => <div className={classNames(props)}>{props.value}</div>;

Badge.defaultProps = {
  isBordered: false,
  levelPoor: 3,
  levelOkay: 6,
  levelGood: 9,
  value: 0
};

Badge.propTypes = {
  isBordered: PropTypes.bool,
  levelPoor: PropTypes.number,
  levelOkay: PropTypes.number,
  levelGood: PropTypes.number,
  value: PropTypes.number
};

export default Badge;
