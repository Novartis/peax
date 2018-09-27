import PropTypes from 'prop-types';
import React from 'react';

// Components
import ButtonIcon from './ButtonIcon';
import DropDownContent from './DropDownContent';
import DropDownTrigger from './DropDownTrigger';
import Icon from './Icon';
import TopBarDropDown from './TopBarDropDown';

// Services
import pubSub from '../services/pub-sub';


class TopBarDropDowLogin extends React.Component {
  constructor(props) {
    super(props);
    this.pubSubs = [];
  }

  componentDidMount() {
    this.pubSubs.push(
      pubSub.subscribe('DropDownTopBarDropDown', this.focusInput.bind(this))
    );
  }

  componentWillUnmount() {
    this.pubSubs.forEach(subscription => pubSub.unsubscribe(subscription));
    this.pubSubs = [];
  }

  focusInput() {
    if (!this.inputEl) return;

    this.inputEl.focus();
  }

  render() {
    return (
      <TopBarDropDown
        alignRight={true}
        className='top-bar-drop-down-login'
        closeOnOuterClick={this.props.closeOnOuterClick}>
        <DropDownTrigger>
          <ButtonIcon className='is-primary-nav' icon='login'>
            Log In
          </ButtonIcon>
        </DropDownTrigger>
        <DropDownContent>
          <form
            className='flex-c flex-v drop-down-form'
            onSubmit={this.props.login}>
            {this.props.isLoginUnsuccessful &&
              <div className='flex-c flex-a-c warning'>
                <Icon iconId='warning' />
                <span>Login failed.</span>
              </div>
            }
            {this.props.isServerUnavailable &&
              <div className='flex-c flex-a-c error'>
                <Icon iconId='warning' />
                <span>Auth server is unavailable.</span>
              </div>
            }
            <input
              placeholder='E-mail or username'
              type='text'
              disabled={this.props.isLoggingIn}
              onChange={this.props.loginUserIdHandler}
              ref={(el) => { this.inputEl = el; }}
              value={this.props.loginUserId} />
            <input
              placeholder='Password'
              type='password'
              disabled={this.props.isLoggingIn}
              onChange={this.props.loginPasswordHandler}
              value={this.props.loginPassword} />
            <button
              type='submit'
              className={`is-primary ${this.props.isLoggingIn ? 'is-active' : ''}`}
              disabled={this.props.isLoggingIn}>
              {this.props.isLoggingIn ? 'Logging in…' : 'Log in'}
            </button>
          </form>
        </DropDownContent>
      </TopBarDropDown>
    );
  }
}

TopBarDropDowLogin.propTypes = {
  closeOnOuterClick: PropTypes.bool,
  isLoggingIn: PropTypes.bool,
  isLoginUnsuccessful: PropTypes.bool,
  isServerUnavailable: PropTypes.bool,
  login: PropTypes.func.isRequired,
  loginPasswordHandler: PropTypes.func.isRequired,
  loginPassword: PropTypes.string.isRequired,
  loginUserIdHandler: PropTypes.func.isRequired,
  loginUserId: PropTypes.string.isRequired,
};

export default TopBarDropDowLogin;
