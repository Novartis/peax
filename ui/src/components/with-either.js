import React from 'react';

const withEither = (conditionalRenderingFn, EitherComponent) => Component => {
  const Either = props =>
    conditionalRenderingFn(props) ? (
      <EitherComponent {...props} />
    ) : (
      <Component {...props} />
    );

  return Either;
};

export default withEither;
