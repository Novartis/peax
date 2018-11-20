// Services
import pubSub from './pub-sub';

// Utils
import getServer from '../utils/get-server';
import cookie from '../utils/cookie';

const state = {
  email: '',
  isAuthenticated: false,
  username: '',
};

const server = getServer();

const checkAuthentication = () => {
  // Get the cookie with the token
  const token = cookie.get('higlasstoken');

  if (!token) { return Promise.resolve(false); }

  // Verify token
  return fetch(
    `${server}/api/v1/current/`,
    {
      // credentials: server === '' ? 'same-origin' : 'include',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      method: 'GET',
    }
  )
    .then(response => response.text()
      .then(data => ({
        data,
        status: response.status,
      })))
    .then((response) => {
      if (response.status !== 200) return false;

      try {
        const data = JSON.parse(response.data);

        state.email = data.email;
        state.isAuthenticated = true;
        state.username = data.username;

        return true;
      } catch (e) {
        // Authentication failed or webserver is broken
        return false;
      }
    })
    .catch(() => {
      state.isAuthenticated = false;
    });
};

const get = (key) => {
  if (key === 'token') return cookie.get('higlasstoken');

  return state[key];
};

const isAuthenticated = () => state.isAuthenticated;

const login = (username, password) => {
  // Remove existing cookie before logging in.
  cookie.remove('higlasstoken');
  state.isAuthenticated = false;

  return fetch(
    `${server}/api-token-auth/`,
    {
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      method: 'POST',
      body: JSON.stringify({
        username,
        password,
      }),
    }
  )
    .then((response) => {
      const contentType = response.headers.get('content-type');

      if (contentType && contentType.indexOf('application/json') !== -1) {
        return response.json().then(body => ({
          isJson: true,
          body,
          response,
        }));
      }

      return response.text().then(body => ({
        isJson: false,
        body,
        response,
      }));
    })
    .then((data) => {
      if (!data.response.ok) {
        throw Error(data.response.statusText);
      }

      state.isAuthenticated = false;

      return data.body;
    })
    .then((data) => {
      cookie.set('higlasstoken', data.token);

      return checkAuthentication();
    })
    .then(() => {
      pubSub.publish('login');
    });
};

const logout = () => {
  // Remove existing cookie before logging in.
  cookie.remove('higlasstoken');

  state.email = '';
  state.isAuthenticated = false;
  state.username = '';

  pubSub.publish('logout');
};

const auth = {
  checkAuthentication,
  get,
  isAuthenticated,
  login,
  logout,
};

export default auth;
