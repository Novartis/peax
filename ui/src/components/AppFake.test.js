import React from 'react';
import ReactDOM from 'react-dom';
import AppFake from './AppFake';

it('renders fake app with spionner', () => {
  const div = document.createElement('div');
  ReactDOM.render(<AppFake/>, div);
});

it('renders fake app with error message', () => {
  const div = document.createElement('div');
  ReactDOM.render(<AppFake error='Something broke. As always&hellip;'/>, div);
});
