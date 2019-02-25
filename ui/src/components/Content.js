import PropTypes from 'prop-types';
import React from 'react';

// Configs
import { RIGHT_BAR_MIN_WIDTH } from './RightBar';

// Styles
import './Content.scss';

const classNames = props => {
  let className = 'flex-g-1 content';

  className += ` ${props.name}-content`;
  className += props.hasRightBar ? ' has-right-bar' : '';
  className += props.hasSubTopBar ? ' has-sub-top-bar' : '';
  className += props.rel ? ' rel' : '';
  className += props.wrap ? ' wrap' : '';
  className += !props.bottomMargin ? ' no-bottom-margin' : '';
  className += props.isVertFlex ? ' flex-c flex-v' : '';
  className += props.hasSmallerTopBar ? ' has-smaller-top-bar' : '';

  return className;
};

const styles = props => {
  if (props.hasRightBar) {
    return {
      marginRight: `${
        props.rightBarShow
          ? Math.max(RIGHT_BAR_MIN_WIDTH, props.rightBarWidth)
          : RIGHT_BAR_MIN_WIDTH
      }px`
    };
  }

  return {};
};

const Content = props => (
  <main className={classNames(props)} style={styles(props)}>
    {props.children}
  </main>
);

Content.defaultProps = {
  bottomMargin: true
};

Content.propTypes = {
  bottomMargin: PropTypes.bool,
  children: PropTypes.node,
  hasRightBar: PropTypes.bool,
  hasSubTopBar: PropTypes.bool,
  isVertFlex: PropTypes.bool,
  hasSmallerTopBar: PropTypes.bool,
  name: PropTypes.string.isRequired,
  rel: PropTypes.bool,
  rightBarShow: PropTypes.bool,
  rightBarWidth: PropTypes.number,
  wrap: PropTypes.bool
};

export default Content;
