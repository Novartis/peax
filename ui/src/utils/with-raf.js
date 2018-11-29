import { requestAnimationFrame } from './request-animation-frame';

const withRaf = (fn, callback) => {
  let isRequesting = false;
  return (...args) => {
    if (isRequesting) {
      return undefined;
    }
    return requestAnimationFrame(() => {
      const resp = fn(...args);
      if (callback) callback(resp);
      isRequesting = false;
    });
  };
};

export default withRaf;
