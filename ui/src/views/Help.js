import PropTypes from 'prop-types';
import React from 'react';

// Higher-order components
import { withPubSub } from '../hocs/pub-sub';

// Components
import Content from '../components/Content';
import ContentWrapper from '../components/ContentWrapper';
import Footer from '../components/Footer';
import Icon from '../components/Icon';
import IconGallery from '../components/IconGallery';

// Utils
import Deferred from '../utils/deferred';

// Stylesheets
import './About.scss';

const showIcons = pubSub => {
  pubSub.publish('globalDialog', {
    message: <IconGallery />,
    request: new Deferred(),
    resolveOnly: true,
    resolveText: 'Close',
    headline: 'All Available Icons'
  });
};

class Help extends React.Component {
  constructor(props) {
    super(props);

    this.pubSubs = [];

    this.swag = [[73, 67, 79, 78, 83]];
    this.swagI = 0;
    this.swagJ = 0;
    this.swagInterval = 500;
    this.swagTime = performance.now();
  }

  componentDidMount() {
    this.pubSubs.push(
      this.props.pubSub.subscribe('keyup', this.keyUpHandler.bind(this))
    );
  }

  componentWillUnmount() {
    this.pubSubs.forEach(subscription =>
      this.props.pubSub.unsubscribe(subscription)
    );
    this.pubSubs = [];
  }

  keyUpHandler(event) {
    this.keyUpSwagHandler(event.keyCode);
  }

  keyUpSwagHandler(keyCode) {
    const now = performance.now();

    if (now - this.swagTime > this.swagInterval) {
      this.swagJ = 0;
    }

    this.swagTime = now;

    if (this.swagJ === 0) {
      this.swag.forEach((codeWurst, index) => {
        if (keyCode === codeWurst[0]) {
          this.swagI = index;
          this.swagJ = 1;
        }
      });
    } else if (keyCode === this.swag[this.swagI][this.swagJ]) {
      this.swagJ += 1;
    }

    if (this.swagJ === this.swag[this.swagI].length) {
      switch (this.swagI) {
        case 0:
          showIcons(this.props.pubSub);
          break;
        default:
        // Nothing
      }
    }
  }

  render() {
    return (
      <ContentWrapper name="help">
        <Content name="help">
          <div className="border-bottom p-t-1 p-b-1">
            <div className="wrap">
              <p>
                You need help getting started with Peax or ran into a tricky
                issue? Fear not! Below is a list of excellent resources that can
                hopefully help you out!
              </p>
            </div>
          </div>

          <div className="wrap p-b-2">
            <h3 id="getting-started" className="iconized underlined anchored">
              <a href="#getting-started" className="hidden-anchor">
                <Icon iconId="link" />
              </a>
              <Icon iconId="launch" />
              <span>Getting Started</span>
            </h3>

            <p>
              At some point in the far far away future you will find some
              marvelous getting started guide here. Until then you are excused
              to freak out.
            </p>

            <h3 id="source-code" className="iconized underlined anchored">
              <a href="#source-code" className="hidden-anchor">
                <Icon iconId="link" />
              </a>
              <Icon iconId="code" />
              Source Code
            </h3>

            <p>Peax uses and adopted the following open source component:</p>

            <ul className="no-list-style large-spacing">
              <li>
                <strong>Genome viewer: </strong>
                <a
                  href="https://github.com/higlass/higlass"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  github.com/higlass/higlass
                </a>
              </li>
              <li>
                <strong>UI architecture: </strong>
                <a
                  href="https://github.com/higlass/higlass-app"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  github.com/higlass/higlass-app
                </a>
              </li>
              <li>
                <strong>Server: </strong>
                <a
                  href="https://github.com/higlass/hgflask"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  github.com/higlass/hgflask
                </a>
              </li>
            </ul>
          </div>
        </Content>
        <Footer />
      </ContentWrapper>
    );
  }
}

Help.propTypes = {
  pubSub: PropTypes.object.isRequired
};

export default withPubSub(Help);
