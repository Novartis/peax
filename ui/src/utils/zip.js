import isSame from './is-same';

class ZipError extends Error {
  constructor(...args) {
    super(...args);
    Error.captureStackTrace(this, ZipError);
  }
}

const zip = (arrays, strides) => {
  if (arrays.length !== strides.length)
    throw new ZipError('number of arrays and number of strides does not match');

  const normLenghts = arrays.map((array, i) => array.length / strides[i]);

  if (!isSame(normLenghts))
    throw new ZipError('normalized array length does not equal');

  const out = [];
  const indices = arrays.map(() => 0);
  const numArrays = arrays.length;

  for (let i = 0; i < normLenghts[0]; i++) {
    const entry = [];
    for (let j = 0; j < numArrays; j++) {
      entry.push(...arrays[j].slice(indices[j], indices[j] + strides[j]));
      indices[j] += strides[j];
    }
    out.push(entry);
  }

  return out;
};

export default zip;
