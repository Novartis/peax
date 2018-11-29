// Custom event Handlers
const resize = pubSub => event => pubSub.publish('resize', event);
const scroll = pubSub => event =>
  pubSub.publish(
    'scrollTop',
    event.target.scrollTop || document.body.scrollTop
  );

/**
 * Supported event handlers.
 *
 * @type {object}
 */
const customEventHandlers = {
  orientationchange: resize,
  scroll
};

/**
 * Get event handler.
 *
 * @param {string} eventName - Name of the event.
 * @return {function} Either a custom or generic event handler.
 */
const getEventHandler = (eventName, pubSub) => {
  if (customEventHandlers[eventName]) {
    return customEventHandlers[eventName](pubSub);
  }
  return event => pubSub.publish(eventName, event);
};

/**
 * Stack of elements with registered event listeners.
 *
 * @type {object}
 */
const registeredEls = {};

/**
 * Unregister an event listener.
 *
 * @param {string} event - Name of the event to stop listening from.
 * @param {object} element - DOM element which we listened to.
 */
const unregister = (event, element) => {
  if (!registeredEls[event] && registeredEls[event] !== element) {
    return;
  }

  registeredEls[event].removeEventListener(event, registeredEls[event].handler);

  registeredEls[event] = undefined;
  delete registeredEls[event];
};

/**
 * Register an event listener.
 *
 * @param {string} event - Name of the event to listen to.
 * @param {object} newElement - DOM element which to listen to.
 */
const register = pubSub => (event, newElement) => {
  if (!newElement || registeredEls[event] === newElement) {
    return;
  }

  if (registeredEls[event]) {
    unregister(registeredEls[event]);
  }

  registeredEls[event] = newElement;
  registeredEls[event].handler = getEventHandler(event, pubSub);
  registeredEls[event].addEventListener(event, registeredEls[event].handler);
};

/**
 * Public API.
 *
 * @type {object}
 */
const createDomEvent = pubSub => ({
  register: register(pubSub),
  unregister
});

export default createDomEvent;
