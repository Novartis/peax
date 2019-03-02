import PropTypes from 'prop-types';
import React from 'react';

// Components
import Button from './Button';
import DropDown from './DropDown';
import DropDownContent from './DropDownContent';
import DropDownTrigger from './DropDownTrigger';

import './DropDownSlider.scss';

const DropDownSlider = props => (
  <DropDown className={`drop-down-slider ${props.reversed ? 'reversed' : ''}`}>
    <DropDownTrigger>
      <Button>{props.value}</Button>
    </DropDownTrigger>
    <DropDownContent>
      <div className="flex-c flex-a-c">
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
      {props.histogram && props.histogram.length && (
        <div className="flex-c histogram">
          {props.histogram.map((bin, i) => (
            <div
              key={i}
              className="flex-g-1"
              style={{
                height: `${Math.min(1, bin / props.histogramNorm) * 100}%`
              }}
            />
          ))}
        </div>
      )}
    </DropDownContent>
  </DropDown>
);

DropDownSlider.defaultProps = {
  disabled: false,
  histogram: [],
  histogramNorm: 1,
  max: 1,
  min: 0,
  reversed: false,
  step: 0.05
};

DropDownSlider.propTypes = {
  disabled: PropTypes.bool,
  histogram: PropTypes.array,
  histogramNorm: PropTypes.number,
  id: PropTypes.string,
  max: PropTypes.number,
  min: PropTypes.number,
  onChange: PropTypes.func.isRequired,
  reversed: PropTypes.bool,
  step: PropTypes.number,
  value: PropTypes.number.isRequired
};

export default DropDownSlider;
