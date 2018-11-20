import PropTypes from 'prop-types';
import React from 'react';
import { withRouter } from 'react-router';
import { NavLink } from 'react-router-dom';

// Components
import Hamburger from './Hamburger';
import Icon from './Icon';
// import TopBarDropDownLogin from './TopBarDropDownLogin';
// import TopBarDropDownUser from './TopBarDropDownUser';

// Services
import auth from '../services/auth';

// Styles
import './TopBar.scss';

const isSearch = pathname => pathname && pathname.match(/\/search(?:(?=.)(\?|\/)|$)/);

class TopBar extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      isLoggingIn: false,
      loginPassword: '',
      loginUserId: '',
      menuIsShown: false,
      userEmail: auth.get('email'),
      userId: auth.get('username'),
    };

    this.loginPasswordHandler = this.loginPasswordHandler.bind(this);
    this.loginUserIdHandler = this.loginUserIdHandler.bind(this);
    this.login = this.login.bind(this);
    this.toggleMenu = this.toggleMenu.bind(this);

    this.unlisten = this.props.history.listen(
      () => this.setState({ menuIsShown: false })
    );
  }

  componentDidUpdate(prevProps) {
    if (prevProps.isAuthenticated !== this.props.isAuthenticated) {
      this.setState({
        isLoggingIn: false,
        userEmail: auth.get('email'),
        userId: auth.get('username'),
      });
    }
  }

  componentWillUnmount() {
    this.unlisten();
  }

  /* ------------------------------ Custom Methods -------------------------- */

  login(event) {
    event.preventDefault();

    this.setState({
      isLoggingIn: true,
      isLoginUnsuccessful: false,
    });

    auth
      .login(this.state.loginUserId, this.state.loginPassword)
      .then((success) => {
        this.setState({
          isLoggingIn: false,
          isLoginUnsuccessful: !success,
          isServerUnavailable: false,
        });
      })
      .catch((error) => {
        if (error.message === 'Bad Request') {
          this.setState({
            isLoggingIn: false,
            isLoginUnsuccessful: true,
            isServerUnavailable: false,
          });
        } else {
          this.setState({
            isLoggingIn: false,
            isServerUnavailable: true,
          });
        }
      });
  }

  loginUserIdHandler(event) {
    this.setState({ loginUserId: event.target.value });
  }

  loginPasswordHandler(event) {
    this.setState({ loginPassword: event.target.value });
  }

  toggleMenu(isOpen) {
    this.setState({
      menuIsShown: isOpen,
    });
  }

  /* -------------------------------- Render -------------------------------- */

  render() {
    return (
      <header className='top-bar'>
        <div className={`flex-c flex-jc-sb top-bar-wrapper ${isSearch(this.props.location.pathname) ? 'wrap-basic' : 'wrap'}`}>
          <div className='flex-c branding-launch'>
            <NavLink to='/' className='flex-c flex-a-c branding'>
              <Icon iconId='logo-two-tone' />
              <span className='higlass'>
                Pea<span className='higlass-hi'>x</span>
              </span>
            </NavLink>
          </div>
          <nav className={`flex-c flex-jc-e flex-a-s is-toggable ${this.state.menuIsShown ? 'is-shown' : ''}`}>
            <ul className='flex-c flex-jc-e flex-a-s no-list-style primary-nav-list'>
              <li>
                <NavLink to='/search' activeClassName='is-active'>Search</NavLink>
              </li>
              <li>
                <NavLink to='/about' activeClassName='is-active'>About</NavLink>
              </li>
              <li>
                <NavLink to='/help' activeClassName='is-active'>Help</NavLink>
              </li>
              {
              // <li className='separated-left flex-c flex-jc-c'>
              //   {this.props.isAuthenticated ?
              //     (
              //       <TopBarDropDownUser
              //         closeOnOuterClick={true}
              //         logout={auth.logout}
              //         userEmail={this.state.userEmail}
              //         userId={this.state.userId} />
              //     ) : (
              //       <TopBarDropDownLogin
              //         closeOnOuterClick={true}
              //         isLoggingIn={this.state.isLoggingIn}
              //         isLoginUnsuccessful={this.state.isLoginUnsuccessful}
              //         isServerUnavailable={this.state.isServerUnavailable}
              //         login={this.login}
              //         loginPassword={this.state.loginPassword}
              //         loginPasswordHandler={this.loginPasswordHandler}
              //         loginUserId={this.state.loginUserId}
              //         loginUserIdHandler={this.loginUserIdHandler} />
              //     )
              //   }
              // </li>
              }
            </ul>
            <Hamburger
              isActive={this.state.menuIsShown}
              onClick={this.toggleMenu} />
          </nav>
        </div>
      </header>
    );
  }
}

TopBar.propTypes = {
  match: PropTypes.object.isRequired,
  location: PropTypes.object.isRequired,
  history: PropTypes.object.isRequired,
  isAuthenticated: PropTypes.bool,
};

export default withRouter(TopBar);
