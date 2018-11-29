import PropTypes from 'prop-types';
import React from 'react';

// Styles
import './Spinner.scss';

const Spinner = props => (
  <div className={`spinner ${props.delayed ? 'is-delayed' : ''}`}>
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 40 40"
      width={props.width}
      height={props.height}
    >
      <circle
        cx="20"
        cy="20"
        r="18"
        strokeWidth="4"
        fill="none"
        stroke="#000"
      />
      <g className="correct" transform="translate(20, 20)">
        <g className="blocks">
          <animateTransform
            attributeName="transform"
            attributeType="XML"
            dur="1.5s"
            from="0"
            repeatCount="indefinite"
            to="360"
            type="rotate"
          />
          <path
            className="one"
            d="M0-20c1.104 0 2 .896 2 2s-.896 2-2 2V0l-4 21h25v-42H0v1z"
            fill="#fff"
          >
            <animateTransform
              attributeName="transform"
              attributeType="XML"
              calcMode="spline"
              dur="1.5s"
              from="0"
              values="0; 360"
              keyTimes="0; 1"
              keySplines="0.2 0.2 0.15 1"
              repeatCount="indefinite"
              to="360"
              type="rotate"
            />
          </path>
          <path
            className="two"
            d="M0-20c-1.104 0-2 .896-2 2s.896 2 2 2V0l4 21h-25v-42H0v1z"
            fill="#fff"
          >
            <animateTransform
              attributeName="transform"
              attributeType="XML"
              calcMode="spline"
              dur="1.5s"
              from="0"
              values="0; 360"
              keyTimes="0; 1"
              keySplines="0.1 0.15 0.8 0.8"
              repeatCount="indefinite"
              to="360"
              type="rotate"
            />
          </path>
        </g>
      </g>
    </svg>
  </div>
);

Spinner.defaultProps = {
  height: 40,
  width: 40
};

Spinner.propTypes = {
  delayed: PropTypes.bool,
  height: PropTypes.oneOfType([PropTypes.number, PropTypes.string]),
  width: PropTypes.oneOfType([PropTypes.number, PropTypes.string])
};

export default Spinner;
