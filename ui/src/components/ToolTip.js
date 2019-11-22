import { PropTypes } from 'prop-types';
import React from 'react';

// Components
import Arrow from './Arrow';

// Styles
import './ToolTip.scss';

class ToolTip extends React.Component {
  constructor(props) {
    super(props);
    this.state = { isShown: false };
    this.hideBound = this.hide.bind(this);
    this.showBound = this.show.bind(this);
  }

  componentWillUnmount() {
    if (this.delayInTimeout) clearTimeout(this.delayInTimeout);
    if (this.delayOutTimeout) clearTimeout(this.delayOutTimeout);
  }

  /* ------------------------------ Custom Methods -------------------------- */

  hide() {
    if (this.delayInTimeout) clearTimeout(this.delayInTimeout);

    this.delayOutTimeout = setTimeout(() => {
      this.setState({ isShown: false });
    }, this.props.delayOut);
  }

  show() {
    if (this.delayOutTimeout) clearTimeout(this.delayOutTimeout);

    this.delayInTimeout = setTimeout(() => {
      this.setState({ isShown: true });
    }, this.props.delayIn);
  }

  /* ---------------------------------- Render ------------------------------ */

  render() {
    let classNames = 'tool-tip';

    classNames += this.state.isShown ? ' is-shown' : '';

    switch (this.props.align) {
      case 'left':
        classNames += ' tool-tip-align-left';
        break;

      case 'right':
        classNames += ' tool-tip-align-right';
        break;

      default:
      // Nothing
    }
    return (
      <div
        className="rel tool-tip-wrapper"
        onMouseEnter={this.showBound}
        onMouseLeave={this.hideBound}
      >
        <div className="tool-tip-anchor">
          <div className={classNames}>
            <Arrow direction="down" size={4} />
            {this.props.title}
          </div>
        </div>
        {this.props.children}
      </div>
    );
  }
}

ToolTip.defaultProps = {
  align: 'center',
  delayIn: 0,
  delayOut: 0
};

ToolTip.propTypes = {
  align: PropTypes.oneOf(['center', 'left', 'right']),
  children: PropTypes.node,
  closeOnClick: PropTypes.bool,
  delayIn: PropTypes.number,
  delayOut: PropTypes.number,
  title: PropTypes.oneOfType([PropTypes.string, PropTypes.node])
};

export default ToolTip;
