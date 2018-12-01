import { isEqual } from 'lodash-es';

import camelToConst from './camel-to-const';
import deepClone from './deep-clone';

const clone = (value, state) => {
  switch (typeof value) {
    case 'object': {
      if (!isEqual(value, state)) {
        return deepClone(value);
      }

      return state;
    }
    default:
      return value;
  }
};

const defaultSetReducer = (key, defaultValue) => (
  state = defaultValue,
  action
) => {
  switch (action.type) {
    case `SET_${camelToConst(key)}`:
      return clone(action.payload[key], state);
    default:
      return state;
  }
};

export default defaultSetReducer;
