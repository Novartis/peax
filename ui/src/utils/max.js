const max = arr => arr.reduce((m, v) => (m >= v ? m : v), -Infinity);

export default max;
