import PropTypes from 'prop-types';
import React from 'react';

// Styles
import './DropNotifier.scss';

class DropNotifier extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      isActive: false
    };

    this.listeners = [];
    this.addEventListeners();
  }

  componentWillUnmount() {
    this.removeEventListeners();
  }

  render() {
    return (
      <div
        className={`drop-notifier flex-c flex-jc-c flex-a-c ${
          this.state.isActive ? 'is-active' : ''
        }`}
      >
        <div
          className="drop-layer full-dim-win"
          ref={el => {
            this.dropLayer = el;
          }}
        />
        <span>Drop JSON Config</span>
      </div>
    );
  }

  /* ------------------------------ Custom Methods -------------------------- */

  addEventListeners() {
    this.eventListeners = [
      {
        name: 'dragenter',
        callback: event => {
          this.setState({
            isActive: true
          });

          event.stopPropagation();
          event.preventDefault();
          return false;
        }
      },
      {
        name: 'dragover',
        callback: event => {
          event.stopPropagation();
          event.preventDefault();
          return false;
        }
      },
      {
        name: 'dragleave',
        callback: event => {
          if (event.target === this.dropLayer) {
            this.setState({
              isActive: false
            });
          }

          event.stopPropagation();
          event.preventDefault();
          return false;
        }
      },
      {
        name: 'drop',
        callback: event => {
          this.setState({
            isActive: false
          });

          this.props.drop(event);

          event.preventDefault();
        }
      }
    ];

    this.eventListeners.forEach(event =>
      document.addEventListener(event.name, event.callback, false)
    );
  }

  removeEventListeners() {
    this.eventListeners.forEach(event =>
      document.removeEventListener(event.name, event.fnc)
    );
  }
}

DropNotifier.propTypes = {
  drop: PropTypes.func.isRequired
};

export default DropNotifier;
