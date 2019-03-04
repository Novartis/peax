import { PropTypes } from 'prop-types';
import React from 'react';

// Utils
import ButtonIcon from './ButtonIcon';

// Styles
import './LabeledSlider.scss';

class LabeledSlider extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      isBam: false
    };

    this.onBamEndBnd = this.onBamEnd.bind(this);
  }

  componentDidMount() {
    this.bamEl.addEventListener('animationend', this.onBamEndBnd);
  }

  componentWillUnmount() {
    this.bamEl.removeEventListener('animationend', this.onBamEndBnd);
  }

  componentDidUpdate(prevProps) {
    if (prevProps.value !== this.props.value) {
      if (this.state.isBam) {
        this.bamEl.style.webkitAnimation = 'none';
        setTimeout(() => {
          this.bamEl.style.webkitAnimation = '';
        }, 0);
      } else {
        this.setState({ isBam: true });
      }
    }
  }

  get valueClass() {
    return this.state.isBam
      ? 'labeled-slider-value labeled-slider-value-changed'
      : 'labeled-slider-value';
  }

  onBamEnd() {
    this.setState({ isBam: false });
  }

  render() {
    let className = 'labeled-slider';

    className += ` ${this.props.sameLine ? ' same-line' : ''}`;
    className += ` ${this.props.className || ''}`;

    return (
      <div className={className}>
        <label className="flex-c flex-jc-sb" htmlFor={this.props.id}>
          <div className="flex-c flex-a-c labeled-slider-label-container">
            <span className="labeled-slider-label">{this.props.label}</span>
            <span className="flex-c flex-jc-c labeled-slider-value-container">
              <span className={this.valueClass}>
                <span
                  ref={ref => {
                    this.bamEl = ref;
                  }}
                  className="labeled-slider-value-bam"
                />
                {this.props.value}
              </span>
            </span>
          </div>
          {this.props.sameLine && (
            <div className="flex-g-1 m-l-0-25">
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
            </div>
          )}
          <div
            className={this.props.info && this.props.sameLine ? 'm-l-0-25' : ''}
          >
            {this.props.info && (
              <ButtonIcon
                className="info-external"
                external={true}
                icon="info"
                iconOnly={true}
                isRound={true}
                href={this.props.info}
              />
            )}
          </div>
        </label>
        {!this.props.sameLine && (
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
        )}
      </div>
    );
  }
}

LabeledSlider.defaultProps = {
  className: '',
  min: 0,
  max: 10,
  onChange: () => {},
  sameLine: false,
  step: 1
};

LabeledSlider.propTypes = {
  className: PropTypes.string,
  disabled: PropTypes.bool,
  id: PropTypes.string,
  info: PropTypes.string,
  label: PropTypes.string.isRequired,
  max: PropTypes.number,
  min: PropTypes.number,
  onChange: PropTypes.func,
  sameLine: PropTypes.bool,
  step: PropTypes.number,
  value: PropTypes.number.isRequired
};

export default LabeledSlider;
