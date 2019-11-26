import { PropTypes } from 'prop-types';
import React from 'react';

// Styles
import './Badge.scss';

const classNames = props => {
  let className = 'badge';

  let level = '';
  if (props.value > props.levelNeutral) level = 'poor';
  if (props.value >= props.levelOkay) level = 'okay';
  if (props.value >= props.levelGood) level = 'good';
  if (!Number.isNaN(+props.valueA) && !Number.isNaN(+props.valueB))
    level = 'pos-neg';

  className += ` badge-level-${level}`;

  if (props.isBordered) className += ' badge-border';

  return className;
};

const Badge = props => (
  <div className={classNames(props)}>
    {!Number.isNaN(+props.valueA) && !Number.isNaN(+props.valueB) ? (
      <div className="flex-c">
        <div className="positive">{props.valueA}</div>
        <div className="negative">{props.valueB}</div>
      </div>
    ) : (
      props.value
    )}
  </div>
);

Badge.defaultProps = {
  isBordered: false,
  levelNeutral: 0,
  levelPoor: 3,
  levelOkay: 6,
  levelGood: 9,
  value: 0
};

Badge.propTypes = {
  isBordered: PropTypes.bool,
  levelNeutral: PropTypes.number,
  levelPoor: PropTypes.number,
  levelOkay: PropTypes.number,
  levelGood: PropTypes.number,
  value: PropTypes.number,
  valueA: PropTypes.number,
  valueB: PropTypes.number
};

export default Badge;
