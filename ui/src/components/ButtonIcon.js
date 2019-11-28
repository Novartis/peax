import { PropTypes } from 'prop-types';
import React from 'react';

// Components
import Icon from './Icon';

// Styles
import './ButtonIcon.scss';

const classNames = props => {
  let className = 'flex-c flex-a-c flex-jc-c button-icon';

  className += ` ${props.className || ''}`;
  className += props.iconOnly ? ' button-icon-only' : '';
  className += props.isActive ? ' is-active' : '';
  className += props.isDisabled ? ' is-disabled' : '';
  className += props.isIconMirrorOnFocus ? ' is-icon-mirror-on-focus' : '';
  className += props.isIconRotationOnFocus ? ' is-icon-rotation-on-focus' : '';
  className += props.iconPosition === 'right' ? ' flex-rev' : '';
  className += props.isRound ? ' is-round' : '';
  className += props.iconSmaller ? ' icon-smaller' : '';
  className += props.smaller ? ' smaller' : '';

  return className;
};

const getTag = href => (href && href.length ? 'a' : 'button');

const ButtonIcon = props => {
  let Tag = getTag(props.href);

  if (!props.href && props.tag) {
    Tag = props.tag;
  }

  return (
    <Tag
      className={classNames(props)}
      href={props.href}
      target={props.external ? '_blank' : ''}
      title={props.title}
      onClick={props.onClick}
      onMouseDown={props.onMouseDown}
      onMouseUp={props.onMouseUp}
      disabled={props.isDisabled}
    >
      {!!props.icon && (
        <Icon
          iconId={props.icon}
          mirrorH={props.iconMirrorH}
          mirrorV={props.iconMirrorV}
        />
      )}
      <span>{props.children}</span>
    </Tag>
  );
};

ButtonIcon.defaultProps = {
  iconPosition: 'left',
  tag: null
};

ButtonIcon.propTypes = {
  children: PropTypes.node,
  className: PropTypes.string,
  external: PropTypes.bool,
  href: PropTypes.string,
  icon: PropTypes.string,
  iconMirrorH: PropTypes.bool,
  iconMirrorV: PropTypes.bool,
  iconOnly: PropTypes.bool,
  iconPosition: PropTypes.oneOf(['left', 'right']),
  iconSmaller: PropTypes.bool,
  isActive: PropTypes.bool,
  isDisabled: PropTypes.bool,
  isIconMirrorOnFocus: PropTypes.bool,
  isIconRotationOnFocus: PropTypes.bool,
  isRound: PropTypes.bool,
  onClick: PropTypes.func,
  onMouseDown: PropTypes.func,
  onMouseUp: PropTypes.func,
  smaller: PropTypes.bool,
  tag: PropTypes.string,
  title: PropTypes.string
};

export default ButtonIcon;
