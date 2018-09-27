const LEVELS = {
  debug: 'log',
  info: 'info',
  warn: 'warn',
  error: 'error',
};

const logger = {
  name: 'Unnamed',
};

Object.keys(LEVELS).forEach((level) => {
  logger[level] = function log(...args) {
    console[LEVELS[level]](`[${level.toUpperCase()}: ${this.name}]`, ...args);  // eslint-disable-line no-console
  };
});

const Logger = name => Object.create(logger, { name: { value: name } });

export default Logger;
