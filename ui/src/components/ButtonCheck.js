import { PropTypes } from 'prop-types';
import React from 'react';

// Components
import Icon from './Icon';

// Styles
import './ButtonCheck.scss';

const classNames = props => {
  let className = 'flex-c flex-a-c flex-jc-c button-check';

  className += props.isActive ? ' is-active' : '';
  className += props.isChecked ? ' is-checked' : '';
  className += props.isDisabled ? ' is-disabled' : '';
  className += props.checkboxPosition === 'right' ? ' flex-rev' : '';
  className += ` ${props.className || ''}`;

  return className;
};

const ButtonIcon = props => (
  <label className={classNames(props)} tabIndex="0">
    <div className="flex-c flex-jc-c flex-a-c checkbox">
      <input
        type="checkbox"
        checked={props.isChecked}
        disabled={props.isDisabled}
        onChange={props.onSelect}
      />
      <Icon iconId="checkmark" />
    </div>
    <span>{props.children}</span>
  </label>
);

ButtonIcon.defaultProps = {
  checkboxPosition: 'left'
};

ButtonIcon.propTypes = {
  checkboxPosition: PropTypes.oneOf(['left', 'right']),
  children: PropTypes.node,
  className: PropTypes.string,
  isActive: PropTypes.bool,
  isChecked: PropTypes.bool,
  isDisabled: PropTypes.bool,
  onSelect: PropTypes.func
};

export default ButtonIcon;
