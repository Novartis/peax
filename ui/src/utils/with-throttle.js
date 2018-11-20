const withThrottle = (delay, fn) => {
  let lastCall = 0;
  return (...args) => {
    const now = Date.now();
    if (now - lastCall < delay) {
      return undefined;
    }
    lastCall = now;
    return fn(...args);
  };
};

export default withThrottle;
