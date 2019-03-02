const objectSet = (obj, path, value) => {
  const branches = [obj];
  const pathArray = typeof path === 'string' ? path.split('.') : path;
  const pathLen = pathArray.length;

  const has = pathArray.every((prop, i) => {
    if (Object.prototype.hasOwnProperty.call(branches[i], prop)) {
      branches[i + 1] = branches[i][prop];
      return true;
    }
    return false;
  });

  if (has) {
    branches[pathLen - 1][pathArray[pathLen - 1]] = value;
  }

  return has;
};

export default objectSet;
