import PropTypes from 'prop-types';
import React from 'react';

// Components
import Content from './Content';
import ContentWrapper from './ContentWrapper';
import MessageCenter from './MessageCenter';
import Icon from './Icon';
import SpinnerCenter from './SpinnerCenter';

// Styles
import './Footer.scss';
import './TopBar.scss';

const AppLoading = props => (
  <div className="app full-mdim">
    <header className="top-bar">
      <div className="flex-c flex-jc-sb top-bar-wrapper wrap">
        <div className="flex-c branding-launch">
          <a href="/" className="flex-c flex-a-c branding">
            <Icon iconId="logo-two-tone" />
            <span className="higlass">
              Pea<span className="higlass-hi">x</span>
            </span>
          </a>
        </div>
        <nav className="flex-c flex-jc-e flex-a-s is-toggable">
          <ul className="flex-c flex-jc-e flex-a-s no-list-style">
            <li>
              <a href="/search">Search</a>
            </li>
            <li>
              <a href="/about">About</a>
            </li>
          </ul>
        </nav>
      </div>
    </header>
    <ContentWrapper name="app-fake">
      <Content name="app-fake" rel={true} wrap={true}>
        {props.error ? (
          <MessageCenter msg={props.error} type="error" />
        ) : (
          <SpinnerCenter delayed={true} />
        )}
      </Content>
      <footer className="footer">
        <div className="wrap flex-c flex-a-c flex-jc-sb">
          <div className="flex-c flex-a-c flex-v">
            <div className="flex-c">
              <Icon
                iconId="logo-seas"
                title="Harvard John A. Paulson School of Engineering and Applied Sciences"
              />
              <Icon
                iconId="logo-novartis"
                title="Novartis Institute for BioMedical Research"
              />
            </div>
          </div>

          <nav>
            <ul className="flex-c flex-jc-e flex-a-s no-list-style">
              <li>
                <a href="/search">Search</a>
              </li>
              <li>
                <a href="/about">About</a>
              </li>
            </ul>
          </nav>
        </div>
      </footer>
    </ContentWrapper>
  </div>
);

AppLoading.propTypes = {
  error: PropTypes.string
};

export default AppLoading;
