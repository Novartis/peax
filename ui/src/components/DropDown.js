import PropTypes from 'prop-types';
import React from 'react';

// Higher-order components
import { withPubSub } from '../hocs/pub-sub';

// Utils
import hasParent from '../utils/has-parent';

class DropDown extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      isOpen: false
    };

    this.pubSubs = [];
  }

  componentDidMount() {
    if (!this.props.noCloseOnOuterClick) {
      this.pubSubs.push(
        this.props.pubSub.subscribe('click', this.clickHandler.bind(this))
      );
    }
  }

  componentWillUnmount() {
    this.pubSubs.forEach(subscription =>
      this.props.pubSub.unsubscribe(subscription)
    );
    this.pubSubs = [];
  }

  componentDidUpdate(prevProps, prevState) {
    if (
      this.props.id &&
      this.state.isOpen &&
      this.state.isOpen !== prevState.isOpen
    ) {
      this.props.pubSub.publish(`DropDown${this.props.id}`, this.state.isOpen);
    }
  }

  render() {
    const childrenWithProps = React.Children.map(this.props.children, child =>
      React.cloneElement(child, {
        dropDownIsOpen: this.state.isOpen,
        dropDownToggle: this.toggle.bind(this)
      })
    );

    let className = 'rel drop-down';

    className += this.props.className ? ` ${this.props.className}` : '';
    className += this.state.isOpen ? ' drop-down-is-open' : '';
    className += this.props.alignRight ? ' drop-down-align-right' : '';
    className += this.props.alignTop ? ' drop-down-align-top' : '';

    return (
      <div
        className={className}
        ref={el => {
          this.el = el;
        }}
      >
        {childrenWithProps}
      </div>
    );
  }

  /* ------------------------------ Custom Methods -------------------------- */

  clickHandler(event) {
    if (!hasParent(event.target, this.el)) {
      this.close();
    }
  }

  close() {
    this.setState({
      isOpen: false
    });
  }

  open() {
    this.setState({
      isOpen: true
    });
  }

  toggle() {
    if (this.state.isOpen) {
      this.close();
    } else {
      this.open();
    }
  }
}

DropDown.defaultProps = {
  alignRight: false,
  alignTop: false,
  className: null,
  id: null,
  noCloseOnOuterClick: false
};

DropDown.propTypes = {
  alignRight: PropTypes.bool,
  alignTop: PropTypes.bool,
  children: PropTypes.node.isRequired,
  className: PropTypes.string,
  id: PropTypes.string,
  noCloseOnOuterClick: PropTypes.bool,
  pubSub: PropTypes.object.isRequired
};

export default withPubSub(DropDown);
