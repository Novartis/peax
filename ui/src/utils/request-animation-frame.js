/**
 * Polyfill-safe method for requesting an animation frame
 *
 * @method  requestAnimationFrame
 * @author  Fritz Lekschas
 * @date    2016-09-12
 * @param   {Function}  callback  Function to be called after a animation frame
 *   has been delivered.
 * @return  {Integer}             ID of the request.
 */
export const requestAnimationFrame = (function requestAnimationFrame() {
  let lastTime = 0;

  return (
    window.requestAnimationFrame ||
    window.webkitRequestAnimationFrame ||
    window.mozRequestAnimationFrame ||
    window.oRequestAnimationFrame ||
    window.msRequestAnimationFrame ||
    function requestAnimationFrameTimeout(callback) {
      const currTime = new Date().getTime();
      const timeToCall = Math.max(0, 16 - (currTime - lastTime));
      const id = window.setTimeout(() => {
        callback(currTime + timeToCall);
      }, timeToCall);
      lastTime = currTime + timeToCall;
      return id;
    }
  );
})();

/**
 * Polyfill-safe method for canceling a requested animation frame
 *
 * @method  cancelAnimationFrame
 * @author  Fritz Lekschas
 * @date    2016-09-12
 * @param   {Integer}  id  ID of the animation frame request to be canceled.
 */
export const cancelAnimationFrame = (function cancelAnimationFrame() {
  return (
    window.cancelAnimationFrame ||
    window.webkitCancelAnimationFrame ||
    window.mozCancelAnimationFrame ||
    window.oCancelAnimationFrame ||
    window.msCancelAnimationFrame ||
    window.cancelAnimationFrame ||
    window.webkitCancelAnimationFrame ||
    window.mozCancelAnimationFrame ||
    window.oCancelAnimationFrame ||
    window.msCancelAnimationFrame ||
    function cancelAnimationFrameTimeout(id) {
      window.clearTimeout(id);
    }
  );
})();

/**
 * Requests the next animation frame.
 *
 * @method  nextAnimationFrame
 * @author  Fritz Lekschas
 * @date    2016-09-12
 * @return  {Object}  Object holding the _request_ and _cancel_ method for
 *   requesting the next animation frame.
 */
const nextAnimationFrame = (function nextAnimationFrame() {
  const ids = {};

  function requestId() {
    let id;
    do {
      id = Math.floor(Math.random() * 1e9);
    } while (id in ids);
    return id;
  }

  return {
    request:
      window.requestNextAnimationFrame ||
      function requestNextAnimationFrame(callback, element) {
        const id = requestId();

        ids[id] = requestAnimationFrame(() => {
          ids[id] = requestAnimationFrame(ts => {
            delete ids[id];
            callback(ts);
          }, element);
        }, element);

        return id;
      },
    cancel:
      window.cancelNextAnimationFrame ||
      function cancelNextAnimationFrame(id) {
        if (ids[id]) {
          cancelAnimationFrame(ids[id]);
          delete ids[id];
        }
      }
  };
})();

export const requestNextAnimationFrame = nextAnimationFrame.request;
export const cancelNextAnimationFrame = nextAnimationFrame.cancel;
