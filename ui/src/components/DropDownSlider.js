import PropTypes from 'prop-types';
import React from 'react';

// Components
import Button from './Button';
import DropDown from './DropDown';
import DropDownContent from './DropDownContent';
import DropDownTrigger from './DropDownTrigger';

import './DropDownSlider.scss';

class DropDownSlider extends React.Component {
  componentDidMount() {}

  componentWillUnmount() {}

  componentDidUpdate() {}

  /* ------------------------------ Custom Methods -------------------------- */

  /* ------------------------------ Custom Methods -------------------------- */

  render() {
    return (
      <DropDown className="drop-down-slider">
        <DropDownTrigger>
          <Button>{this.props.value}</Button>
        </DropDownTrigger>
        <DropDownContent>
          <div className="flex-c flex-a-c">
            <span className="drop-down-slider-min">{this.props.min}</span>
            <input
              id={this.props.id}
              type="range"
              min={this.props.min}
              max={this.props.max}
              step={this.props.step}
              disabled={this.props.disabled}
              value={this.props.value}
              onChange={this.props.onChange}
            />
            <span className="drop-down-slider-max">{this.props.max}</span>
          </div>
        </DropDownContent>
      </DropDown>
    );
  }
}

DropDownSlider.defaultProps = {
  min: 0,
  max: 1,
  step: 0.05,
  disabled: false
};

DropDownSlider.propTypes = {
  id: PropTypes.string,
  min: PropTypes.number,
  max: PropTypes.number,
  step: PropTypes.number,
  value: PropTypes.number.isRequired,
  disabled: PropTypes.bool,
  onChange: PropTypes.func.isRequired
};

export default DropDownSlider;
