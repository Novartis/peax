import { PropTypes } from 'prop-types';
import React from 'react';

// Styles
import './Arrow.scss';

const classNames = (props) => {
  let className = 'arrow';

  className += ` arrow-${props.direction}`;

  return className;
};

const styles = props => ({
  borderTopColor: `${props.direction === 'down' ? props.color : 'transparent'}`,
  borderTopWidth: `${props.direction !== 'up' ? props.size : 0}px`,
  borderRightColor: `${props.direction === 'left' ? props.color : 'transparent'}`,
  borderRightWidth: `${props.direction !== 'right' ? props.size : 0}px`,
  borderBottomColor: `${props.direction === 'up' ? props.color : 'transparent'}`,
  borderBottomWidth: `${props.direction !== 'down' ? props.size : 0}px`,
  borderLeftColor: `${props.direction === 'right' ? props.color : 'transparent'}`,
  borderLeftWidth: `${props.direction !== 'left' ? props.size : 0}px`,
});

const Arrow = props => (
  <div
    className={classNames(props)}
    style={styles(props)} />
);

Arrow.defaultProps = {
  color: '#000',
  direction: 'top',
  size: 5,
};

Arrow.propTypes = {
  color: PropTypes.string,
  direction: PropTypes.oneOf(['up', 'right', 'down', 'left']),
  size: PropTypes.number,
};

export default Arrow;
