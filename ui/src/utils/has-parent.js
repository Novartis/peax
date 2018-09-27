/**
 * Check if an element is the child of another element.
 *
 * @param {object} source - Source element to be checked for if `target` is a
 *   parent.
 * @param {object} target - Target, i.e., parent, element.
 * @return {boolean} If `true` `target` is a parent of `source`.
 */
const hasParent = (source, target) => {
  let currentEl = source;

  while (currentEl && currentEl !== target && currentEl.tagName !== 'HTML') {
    currentEl = currentEl.parentNode;
  }

  if (currentEl === target) {
    return true;
  }

  return false;
};

export default hasParent;
