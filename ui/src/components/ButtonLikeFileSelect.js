import { PropTypes } from 'prop-types';
import React from 'react';

// Components
import Icon from './Icon';

// Styles
import './ButtonLikeFileSelect.scss';

let inputEl;

const ButtonLikeFileSelect = props => (
  <div
    className={`flex-c flex-a-c button-like-file-select ${props.className}`}
    onClick={() => inputEl.click()}
  >
    <span className="flex-g-1 button-like-select-text">{props.children}</span>
    <Icon iconId="arrow-bottom" />
    <input
      type="file"
      accept=".json"
      ref={el => {
        inputEl = el;
      }}
      onChange={props.select}
    />
  </div>
);

ButtonLikeFileSelect.defaultProps = {
  className: ''
};

ButtonLikeFileSelect.propTypes = {
  children: PropTypes.node.isRequired,
  className: PropTypes.string,
  select: PropTypes.func.isRequired
};

export default ButtonLikeFileSelect;
