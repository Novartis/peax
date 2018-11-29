import PropTypes from 'prop-types';
import React from 'react';

// Higher-order components
import { withPubSub } from '../hocs/pub-sub';

// Components
import ButtonIcon from './ButtonIcon';

// Styles
import './RightBar.scss';

export const RIGHT_BAR_MIN_WIDTH = 4;

class RightBar extends React.Component {
  constructor(props) {
    super(props);

    this.pubSubs = [];
  }

  componentDidMount() {
    this.pubSubs.push(
      this.props.pubSub.subscribe('mousemove', this.mouseMoveHandler.bind(this))
    );
    this.pubSubs.push(
      this.props.pubSub.subscribe('mouseup', this.mouseUpHandler.bind(this))
    );
  }

  componentWillUnmount() {
    this.pubSubs.forEach(subscription =>
      this.props.pubSub.unsubscribe(subscription)
    );
    this.pubSubs = [];
  }

  render() {
    let classNames = 'right-bar';
    classNames += this.props.isShown ? ' is-shown' : '';
    classNames += this.props.className ? ` ${this.props.className}` : '';

    const styles = {
      width: `${
        this.props.isShown
          ? Math.max(RIGHT_BAR_MIN_WIDTH, this.props.width)
          : RIGHT_BAR_MIN_WIDTH
      }px`
    };

    return (
      <aside
        className={classNames}
        style={styles}
        onTransitionEnd={this.transitionListener.bind(this)}
      >
        <ButtonIcon
          className="right-bar-toggler"
          icon="arrow-right-double"
          iconMirrorV={!this.props.isShown}
          iconOnly={true}
          onMouseDown={this.mouseDownHandler.bind(this)}
        />
        <div className="right-bar-container">{this.props.children}</div>
      </aside>
    );
  }

  /* ------------------------------ Custom Methods -------------------------- */

  mouseDownHandler(event) {
    this.mouseDown = true;
    this.mouseDownTime = performance.now();
    this.mouseDownX = event.clientX;

    if (this.props.isShown) {
      this.mouseDownWidth = this.props.width;
    } else {
      this.mouseDownWidth = 0;
    }
  }

  mouseMoveHandler(event) {
    if (this.mouseDown) {
      const deltaX = this.mouseDownX - event.clientX;
      this.props.widthSetter(this.mouseDownWidth + deltaX);
      if (!this.props.isShown) {
        this.props.show(true);
      }
    }
  }

  mouseUpHandler(event) {
    if (this.mouseDown) {
      const deltaTime = performance.now() - this.mouseDownTime;
      const deltaX = this.mouseDownX - event.clientX;

      if (Math.abs(deltaX) < 2 && deltaTime < 150) {
        this.callWidthSetterFinal = true;
        this.props.toggle();
      } else {
        this.props.widthSetter(this.mouseDownWidth + deltaX);
        this.props.widthSetterFinal();
      }

      this.mouseDown = false;
    }
  }

  transitionListener() {
    if (this.callWidthSetterFinal) {
      this.props.widthSetterFinal();
      this.callWidthSetterFinal = false;
    }
  }
}

RightBar.defaultProps = {
  className: ''
};

RightBar.propTypes = {
  children: PropTypes.node.isRequired,
  className: PropTypes.string,
  isShown: PropTypes.bool,
  pubSub: PropTypes.object.isRequired,
  show: PropTypes.func,
  toggle: PropTypes.func,
  width: PropTypes.number,
  widthSetter: PropTypes.func,
  widthSetterFinal: PropTypes.func
};

export default withPubSub(RightBar);
