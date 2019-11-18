import PropTypes from 'prop-types';
import React from 'react';

// Components
import Badge from './Badge';
import Button from './Button';
import ButtonIcon from './ButtonIcon';
import DropDown from './DropDown';
import DropDownContent from './DropDownContent';
import DropDownTrigger from './DropDownTrigger';

import './DropDownMenu.scss';

const DropDownMenu = props => (
  <DropDown className={`drop-down-menu ${props.className}`}>
    <DropDownTrigger>
      {props.triggerIcon ? (
        <ButtonIcon icon={props.triggerIcon}>{props.trigger}</ButtonIcon>
      ) : (
        <Button>{props.trigger}</Button>
      )}
    </DropDownTrigger>
    <DropDownContent>
      <ul className="no-list-style flex-c flex-v">
        {props.items.map((item, i) => (
          <Button
            key={i}
            tag="li"
            className="drop-down-menu-item"
            onClick={item.onClick}
          >
            <div className="flex-c flex-jc-sb">
              <strong>{item.name}</strong>
              <Badge
                levelPoor={1}
                levelOkay={Infinity}
                levelGood={Infinity}
                value={item.getNumber()}
              />
            </div>
            <p className="description">{item.description}</p>
          </Button>
        ))}
      </ul>
    </DropDownContent>
  </DropDown>
);

DropDownMenu.defaultProps = {
  disabled: false
};

DropDownMenu.propTypes = {
  className: PropTypes.string,
  disabled: PropTypes.bool,
  id: PropTypes.string,
  items: PropTypes.array.isRequired,
  trigger: PropTypes.string.isRequired,
  triggerIcon: PropTypes.string
};

export default DropDownMenu;
