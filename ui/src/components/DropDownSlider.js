import PropTypes from 'prop-types';
import React from 'react';

// Components
import Button from './Button';
import DropDown from './DropDown';
import DropDownContent from './DropDownContent';
import DropDownTrigger from './DropDownTrigger';

import './DropDownSlider.scss';

const DropDownSlider = props => (
  <DropDown className="drop-down-slider">
    <DropDownTrigger>
      <Button>{props.value}</Button>
    </DropDownTrigger>
    <DropDownContent>
      <div className={`flex-c flex-a-c ${props.reversed ? 'reversed' : ''}`}>
        {props.reversed ? (
          <span className="drop-down-slider-max">{props.max}</span>
        ) : (
          <span className="drop-down-slider-min">{props.min}</span>
        )}
        <input
          id={props.id}
          type="range"
          min={props.min}
          max={props.max}
          step={props.step}
          disabled={props.disabled}
          value={props.value}
          onChange={props.onChange}
        />
        {props.reversed ? (
          <span className="drop-down-slider-min">{props.min}</span>
        ) : (
          <span className="drop-down-slider-max">{props.max}</span>
        )}
      </div>
    </DropDownContent>
  </DropDown>
);

DropDownSlider.defaultProps = {
  disabled: false,
  max: 1,
  min: 0,
  reversed: false,
  step: 0.05
};

DropDownSlider.propTypes = {
  disabled: PropTypes.bool,
  id: PropTypes.string,
  max: PropTypes.number,
  min: PropTypes.number,
  onChange: PropTypes.func.isRequired,
  reversed: PropTypes.bool,
  step: PropTypes.number,
  value: PropTypes.number.isRequired
};

export default DropDownSlider;
