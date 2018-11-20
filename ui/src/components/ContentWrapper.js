import PropTypes from 'prop-types';
import React from 'react';

// Components
import ErrorBar from './ErrorBar';

// Services
import pubSub from '../services/pub-sub';

class ContentWrapper extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      error: '',
    };

    this.pubSubs = [];
  }

  componentDidMount() {
    this.pubSubs.push(
      pubSub.subscribe('globalError', this.errorHandler.bind(this))
    );
  }

  componentWillUnmount() {
    this.pubSubs.forEach(subscription => pubSub.unsubscribe(subscription));
  }

  render() {
    let className = `flex-c flex-v full-mdim content-wrapper ${this.props.name}`;

    className += this.props.isFullDimOnly ? ' oh' : '';
    className += this.props.bottomBar ? ' content-wrapper-bottom-bar' : '';
    className += this.state.error ? ' content-wrapper-has-error' : '';

    return (
      <div className={className}>
        {this.state.error &&
          <ErrorBar
            autoClose={true}
            isClosable={true}
            msg={this.state.error}
            onClose={() => this.setState({ error: '' })}
            wrap={!this.props.isFullDimOnly}
          />
        }
        {this.props.children}
      </div>
    );
  }

  /* ------------------------------ Custom Methods -------------------------- */

  errorHandler(error) {
    this.setState({
      error,
    });
  }
}

ContentWrapper.defaultProps = {
  bottomBar: false,
};

ContentWrapper.propTypes = {
  bottomBar: PropTypes.bool,
  children: PropTypes.node,
  isFullDimOnly: PropTypes.bool,
  name: PropTypes.string.isRequired,
};

export default ContentWrapper;
