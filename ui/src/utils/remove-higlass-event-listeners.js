export const removeHiGlassEventListeners = (listeners, api) => {
  listeners = !Array.isArray(listeners) ? Object.values(listeners) : listeners; // eslint-disable-line no-param-reassign
  listeners.forEach(listener => {
    api.off(listener.event, listener.id);
  });
  return [];
};

export default removeHiGlassEventListeners;
