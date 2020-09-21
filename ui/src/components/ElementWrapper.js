import PropTypes from 'prop-types';
import React from 'react';

import { toVoid } from '../utils';

class ElementWrapper extends React.Component {
  render() {
    return (
      <div
        className={this.props.className}
        ref={el => {
          if (el && this.props.element) {
            el.appendChild(this.props.element);
            this.props.onAppend();
          }
        }}
      />
    );
  }
}

ElementWrapper.defaultProps = {
  className: null,
  onAppend: toVoid
};

ElementWrapper.propTypes = {
  className: PropTypes.string,
  element: PropTypes.object,
  onAppend: PropTypes.func
};

export default ElementWrapper;
