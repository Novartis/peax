import PropTypes from 'prop-types';
import React from 'react';

class ElementWrapper extends React.Component {
  render() {
    return (
      <div
        className={this.props.className}
        ref={el => {
          if (el && this.props.element) el.appendChild(this.props.element);
        }}
      />
    );
  }
}

ElementWrapper.defaultProps = {
  className: null
};

ElementWrapper.propTypes = {
  className: PropTypes.string,
  element: PropTypes.object
};

export default ElementWrapper;
