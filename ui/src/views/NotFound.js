import PropTypes from 'prop-types';
import React from 'react';

// Components
import Content from '../components/Content';
import ContentWrapper from '../components/ContentWrapper';
import Footer from '../components/Footer';
import Icon from '../components/Icon';

import './NotFound.scss';

const NotFound = props => (
  <ContentWrapper name="not-found">
    <Content name="not-found" wrap={true} rel={true}>
      <div className="flex-c flex-v flex-a-c flex-jc-c full-dim">
        <div className="flex-c flex-v flex-a-c not-found-header">
          <div className="icon-wrapper">
            <Icon iconId="sad" />
          </div>
          <h2 className="m-t-0">{props.title}</h2>
        </div>
        <em>{props.message}</em>
      </div>
    </Content>
    <Footer />
  </ContentWrapper>
);

NotFound.defaultProps = {
  message: 'The requested page either moved or does not exist.',
  title: 'Nothing Found!'
};

NotFound.propTypes = {
  message: PropTypes.string,
  title: PropTypes.string
};

export default NotFound;
