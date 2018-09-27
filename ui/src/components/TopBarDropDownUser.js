import PropTypes from 'prop-types';
import React from 'react';

// Components
import ButtonIcon from './ButtonIcon';
import DropDownContent from './DropDownContent';
import DropDownTrigger from './DropDownTrigger';
import TopBarDropDown from './TopBarDropDown';

// Styles
import './TopBarDropDownUser.scss';


const TopBarDropDowUser = props => (
  <TopBarDropDown
    alignRight={true}
    className='top-bar-drop-down-user'
    closeOnOuterClick={props.closeOnOuterClick}>
    <DropDownTrigger>
      <ButtonIcon className='is-primary-nav' icon='person' iconOnly={true} />
    </DropDownTrigger>
    <DropDownContent>
      <div className='flex-c flex-v'>
        <div className='menu-field menu-text'>
          <div className='menu-text-label'>Signed in as:</div>
          <div>{props.userId}</div>
        </div>
        <div className='menu-separator' />
        <button
          className='menu-field menu-button'
          onClick={props.logout}>Sign out</button>
      </div>
    </DropDownContent>
  </TopBarDropDown>
);

TopBarDropDowUser.propTypes = {
  closeOnOuterClick: PropTypes.bool,
  logout: PropTypes.func.isRequired,
  userEmail: PropTypes.string.isRequired,
  userId: PropTypes.string.isRequired,
};

export default TopBarDropDowUser;
