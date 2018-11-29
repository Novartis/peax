import { PropTypes } from 'prop-types';
import React from 'react';

// Components
import Icon from './Icon';

// Styles
import './InputIcon.scss';

const classNames = props => {
  let className = 'input-icon';

  className += ` ${props.className}`;
  className += props.isActive ? ' is-active' : '';
  className += props.isDisabled ? ' is-disabled' : '';

  return className;
};

const InputIcon = props => (
  <div className={classNames(props)}>
    <Icon
      iconId={props.icon}
      mirrorH={props.iconMirrorH}
      mirrorV={props.iconMirrorV}
    />
    <input
      type={props.type}
      disabled={props.isDisabled}
      placeholder={props.placeholder}
      onChange={props.onChange}
      onInput={props.onInput}
    />
  </div>
);

InputIcon.defaultProps = {
  isDisabled: false,
  placeholder: '',
  type: 'text'
};

InputIcon.propTypes = {
  className: PropTypes.string,
  icon: PropTypes.string.isRequired,
  iconMirrorH: PropTypes.bool,
  iconMirrorV: PropTypes.bool,
  isActive: PropTypes.bool,
  isDisabled: PropTypes.bool,
  onChange: PropTypes.func,
  onInput: PropTypes.func,
  type: PropTypes.string,
  placeholder: PropTypes.string
};

export default InputIcon;
