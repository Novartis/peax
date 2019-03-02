const objectHas = (obj, path) => {
  let branch = obj;

  return (typeof path === 'string' ? path.split('.') : path).every(prop => {
    if (Object.prototype.hasOwnProperty.call(branch, prop)) {
      branch = branch[prop];
      return true;
    }
    return false;
  });
};

export default objectHas;
