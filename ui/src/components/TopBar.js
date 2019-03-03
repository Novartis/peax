import PropTypes from 'prop-types';
import React from 'react';
import { withRouter } from 'react-router';
import { NavLink } from 'react-router-dom';

// Components
import Hamburger from './Hamburger';
import Icon from './Icon';

// Styles
import './TopBar.scss';

const isSearch = pathname =>
  pathname && pathname.match(/\/search(?:(?=.)(\?|\/)|$)/);

class TopBar extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      menuIsShown: false
    };

    this.toggleMenu = this.toggleMenu.bind(this);

    this.unlisten = this.props.history.listen(() =>
      this.setState({ menuIsShown: false })
    );
  }

  componentWillUnmount() {
    this.unlisten();
  }

  /* ------------------------------ Custom Methods -------------------------- */

  toggleMenu(isOpen) {
    this.setState({
      menuIsShown: isOpen
    });
  }

  /* -------------------------------- Render -------------------------------- */

  render() {
    let wrapClass = 'wrap';
    let sizeClass = '';

    if (isSearch(this.props.location.pathname)) {
      wrapClass = 'wrap-basic';
      sizeClass = 'smaller';
    }

    return (
      <header className={`top-bar ${sizeClass}`}>
        <div className={`flex-c flex-jc-sb top-bar-wrapper ${wrapClass}`}>
          <div className="flex-c branding-launch">
            <NavLink to="/" className="flex-c flex-a-c branding">
              <Icon iconId="logo-two-tone" />
              <span className="higlass">
                Pea<span className="higlass-hi">x</span>
              </span>
            </NavLink>
          </div>
          <nav
            className={`flex-c flex-jc-e flex-a-s is-toggable ${
              this.state.menuIsShown ? 'is-shown' : ''
            }`}
          >
            <ul className="flex-c flex-jc-e flex-a-s no-list-style primary-nav-list">
              <li>
                <NavLink to="/search" activeClassName="is-active">
                  Searches
                </NavLink>
              </li>
              <li>
                <NavLink to="/about" activeClassName="is-active">
                  About
                </NavLink>
              </li>
              {
                // <li>
                //   <NavLink to="/help" activeClassName="is-active">
                //     Help
                //   </NavLink>
                // </li>
              }
            </ul>
            <Hamburger
              isActive={this.state.menuIsShown}
              onClick={this.toggleMenu}
            />
          </nav>
        </div>
      </header>
    );
  }
}

TopBar.propTypes = {
  match: PropTypes.object.isRequired,
  location: PropTypes.object.isRequired,
  history: PropTypes.object.isRequired
};

export default withRouter(TopBar);
