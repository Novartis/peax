import { PropTypes } from "prop-types";
import React from "react";

// Styles
import "./Button.scss";

const classNames = props => {
  let className = "button";

  className += ` ${props.className}`;
  className += props.isActive ? " is-active" : "";
  className += props.isDisabled ? " is-disabled" : "";

  return className;
};

const Button = props => (
  <button
    className={classNames(props)}
    title={props.title}
    onClick={props.onClick}
    disabled={props.isDisabled}
  >
    {props.children}
  </button>
);

Button.defaultProps = {
  className: ""
};

Button.propTypes = {
  children: PropTypes.node,
  className: PropTypes.string,
  isActive: PropTypes.bool,
  isDisabled: PropTypes.bool,
  onClick: PropTypes.func,
  title: PropTypes.string
};

export default Button;
