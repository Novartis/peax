import { PropTypes } from 'prop-types';
import React from 'react';

// Components
import ButtonIcon from './ButtonIcon';

// Styles
import './ButtonRadio.scss';

const classNames = props => {
  let className = 'flex-c flex-a-c flex-jc-c button-radio';

  className += props.isVertical ? ' flex-v' : '';
  className += props.isActive ? ' is-active' : '';
  className += props.isDisabled ? ' is-disabled' : '';
  className += props.selection ? ' is-selected' : '';
  className += ` ${props.className}`;

  return className;
};

const classNamesLabel = props => {
  let className = 'flex-g-1 flex-c flex-a-c flex-jc-c button-radio-wrapper';

  className += props.isVertical ? ' full-w' : ' full-h';

  return className;
};

const onClickHandler = (props, value, deselectedValue) => {
  if (!props.isDisabled) props.onClick(value, deselectedValue);
};

const isActive = (props, value) => {
  if (
    props.defaultSelection === value &&
    ((props.isMultiple && !props.selection.some(s => props.options[s])) ||
      !props.options[props.selection])
  )
    return true;

  if (props.isMultiple) return props.selection.includes(value);

  return props.selection === value;
};

const ButtonRadio = props => (
  <div className={classNames(props)}>
    {props.label && <label>{props.label}</label>}
    {Object.values(props.options).map(option => (
      <div key={option.value} className={classNamesLabel(props)}>
        <ButtonIcon
          className={option.value}
          icon={option.icon}
          iconOnly={option.iconOnly}
          iconMirrorH={option.iconMirrorH}
          iconMirrorV={option.iconMirrorV}
          isActive={isActive(props, option.value)}
          isDisabled={props.isDisabled}
          onClick={() => {
            if (props.isDeselectable && isActive(props, option.value)) {
              onClickHandler(props, null, option.value);
            } else {
              onClickHandler(props, option.value, null);
            }
          }}
        >
          {option.name}
        </ButtonIcon>
      </div>
    ))}
  </div>
);

ButtonRadio.defaultProps = {
  className: ''
};

ButtonRadio.propTypes = {
  className: PropTypes.string,
  defaultSelection: PropTypes.string,
  isActive: PropTypes.bool,
  isDeselectable: PropTypes.bool,
  isDisabled: PropTypes.bool,
  isMultiple: PropTypes.bool,
  isVertical: PropTypes.bool,
  label: PropTypes.string,
  name: PropTypes.string,
  onClick: PropTypes.func,
  options: PropTypes.objectOf(
    PropTypes.shape({
      icon: PropTypes.string,
      iconMirrorH: PropTypes.bool,
      iconMirrorV: PropTypes.bool,
      iconOnly: PropTypes.bool,
      name: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
      value: PropTypes.oneOfType([PropTypes.number, PropTypes.string])
    })
  ),
  selection: PropTypes.oneOfType([PropTypes.string, PropTypes.array])
};

export default ButtonRadio;
