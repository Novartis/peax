import React from 'react';

const withMaybe = conditionalRenderingFn => Component => {
  const Maybe = props =>
    conditionalRenderingFn(props) ? null : <Component {...props} />;

  return Maybe;
};

export default withMaybe;
