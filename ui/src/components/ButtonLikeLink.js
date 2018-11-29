import { PropTypes } from 'prop-types';
import React from 'react';
import { Link } from 'react-router-dom';

// Styles
import './ButtonLikeLink.scss';

const ButtonLikeLink = props => (
  <div className={`flex-c flex-a-c button-like-link ${props.className}`}>
    <Link to={props.to}>{props.children}</Link>
  </div>
);

ButtonLikeLink.defaultProps = {
  className: ''
};

ButtonLikeLink.propTypes = {
  children: PropTypes.node.isRequired,
  className: PropTypes.string,
  to: PropTypes.string.isRequired
};

export default ButtonLikeLink;
