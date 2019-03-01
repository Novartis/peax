import PropTypes from 'prop-types';
import React from 'react';

// Components
import Icon from './Icon';

// Styles
import './TabEntry.scss';

class TabEntry extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      isOpen: true
    };
  }

  render() {
    let className = 'tab-entry';

    className += this.props.className ? ` ${this.props.className}` : '';
    className +=
      this.isOpen && this.props.isHeightStretching ? ' flex-g-1' : '';
    className += this.props.title ? ' tab-entry-has-header' : '';
    className += this.props.isVScrollable ? ' is-v-scrollable' : '';

    return (
      <div className={className}>
        {this.props.title && (
          <div
            className="tab-entry-header flex-c flex-a-c"
            onClick={this.toggle.bind(this)}
          >
            <Icon iconId="arrow-bottom" mirrorH={!this.isOpen} />
            {this.props.title}
          </div>
        )}
        <div className="tab-entry-content">
          {this.isOpen && this.props.children}
        </div>
      </div>
    );
  }

  /* ----------------------------- Getter / Setter -------------------------- */

  get isOpen() {
    return typeof this.props.isOpen !== 'undefined'
      ? this.props.isOpen
      : this.state.isOpen;
  }

  /* ------------------------------ Custom Methods -------------------------- */

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
    if (typeof this.props.isOpen !== 'undefined' && this.props.toggle) {
      this.props.toggle(this.props.isOpen);
      return;
    }

    if (this.isOpen) {
      this.close();
    } else {
      this.open();
    }
  }
}

TabEntry.propTypes = {
  children: PropTypes.node.isRequired,
  className: PropTypes.string,
  isCollapsible: PropTypes.bool,
  isHeightStretching: PropTypes.bool,
  isOpen: PropTypes.bool,
  isVScrollable: PropTypes.bool,
  title: PropTypes.string,
  toggle: PropTypes.func
};

export default TabEntry;
