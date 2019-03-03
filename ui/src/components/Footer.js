import React from 'react';
import { NavLink } from 'react-router-dom';

// Components
import Icon from './Icon';
import ToolTip from './ToolTip';

// Styles
import './Footer.scss';

const Footer = () => (
  <footer className="footer">
    <div className="wrap flex-c flex-a-c flex-jc-sb">
      <div className="flex-c flex-a-c flex-v">
        <div className="flex-c">
          <ToolTip
            align="left"
            delayIn={2000}
            delayOut={500}
            title={
              <span className="flex-c">
                <span>
                  Harvard John A. Paulson School of Engineering and Applied
                  Sciences
                </span>
              </span>
            }
          >
            <Icon iconId="logo-seas" />
          </ToolTip>
          <ToolTip
            align="left"
            delayIn={2000}
            delayOut={500}
            title={
              <span className="flex-c">
                <span>Novartis Institute of BioMedical Research</span>
              </span>
            }
          >
            <Icon iconId="logo-novartis" />
          </ToolTip>
        </div>
      </div>

      <nav>
        <ul className="flex-c flex-jc-e flex-a-s no-list-style">
          <li>
            <NavLink exact to="/" activeClassName="is-active">
              Home
            </NavLink>
          </li>
          <li>
            <NavLink exact to="/search" activeClassName="is-active">
              Searches
            </NavLink>
          </li>
          <li>
            <NavLink exact to="/about" activeClassName="is-active">
              About
            </NavLink>
          </li>
          {
            // <li>
            //   <NavLink exact to="/help" activeClassName="is-active">
            //     Help
            //   </NavLink>
            // </li>
          }
        </ul>
      </nav>
    </div>
  </footer>
);

export default Footer;
