import React from 'react';

const { Provider, Consumer } = React.createContext();

// Higher order component
const withPubSub = Component =>
  React.forwardRef((props, ref) => (
    <Consumer>
      {pubSub => <Component ref={ref} {...props} pubSub={pubSub} />}
    </Consumer>
  ));

export default withPubSub;

export { Provider, withPubSub };
