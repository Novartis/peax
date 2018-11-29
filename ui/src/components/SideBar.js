import PropTypes from 'prop-types';
import React from 'react';

// Higher-order components
import { withPubSub } from '../hocs/pub-sub';

// Utils
import debounce from '../utils/debounce';

// Styles
import './SideBar.scss';

class SideBar extends React.Component {
  constructor(props) {
    super(props);

    this.checkStickAbilityDb = debounce(this.checkStickAbility.bind(this), 50);
    this.scrollHandlerDb = debounce(this.scrollHandler.bind(this), 50);

    this.sidebarOffsetTop = 0;
    this.state = {
      style: {
        marginTop: 0
      }
    };

    this.pubSubs = [];
  }

  componentDidMount() {
    if (this.props.isSticky) {
      this.checkStickAbility();

      this.sidebarOffsetTop =
        this.sideBarEl.getBoundingClientRect().top -
        document.body.getBoundingClientRect().top;

      this.pubSubs.push(
        this.props.pubSub.subscribe('resize', this.checkStickAbilityDb)
      );
      this.pubSubs.push(
        this.props.pubSub.subscribe('scrollTop', this.scrollHandlerDb)
      );
    }
  }

  componentWillUnmount() {
    this.pubSubs.forEach(subscription =>
      this.props.pubSub.unsubscribe(subscription)
    );
    this.pubSubs = [];
  }

  render() {
    return (
      <aside
        className="side-bar"
        ref={el => {
          this.sideBarEl = el;
        }}
        style={this.state.style}
      >
        {this.props.children}
      </aside>
    );
  }

  /* ---------------------------- Custom Methods ---------------------------- */

  checkStickAbility() {
    if (!this.sideBarEl) {
      this.stickinessDisabled = true;
      return;
    }

    // Header = 3rem; Margin = 1rem; 1rem = 16px
    const height = this.sideBarEl.getBoundingClientRect().height + 16 * 4;

    this.stickinessDisabled = window.innerHeight < height;
  }

  scrollHandler(scrollTop) {
    if (!this.stickinessDisabled) {
      this.setState({
        style: {
          marginTop: `${scrollTop}px`
        }
      });
    } else if (
      this.state.style.marginTop !== 0 ||
      this.state.style.marginTop !== '0px'
    ) {
      this.setState({
        style: {
          marginTop: 0
        }
      });
    }
  }
}

SideBar.defaultProps = {
  isSticky: false
};

SideBar.propTypes = {
  children: PropTypes.node.isRequired,
  pubSub: PropTypes.object.isRequired,
  isSticky: PropTypes.bool
};

export default withPubSub(SideBar);
