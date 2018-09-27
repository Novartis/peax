/**
 * Debounce a function call.
 *
 * @description
 * Function calls are delayed by `wait` milliseconds and only one out of
 * multiple function calls is executed.
 *
 * @param {functiom} func - Function to be debounced
 * @param {number} wait - Number of milliseconds to debounce the function call.
 * @param {boolean} immediate - If `true` function is not debounced.
 * @return {functiomn} Debounced function.
 */
const debounce = (func, wait, immediate) => {
  let timeout;

  const debounced = (...args) => {
    const later = () => {
      timeout = null;
      if (!immediate) {
        func(...args);
      }
    };

    const callNow = immediate && !timeout;
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);

    if (callNow) {
      func(...args);
    }
  };

  debounce.cancel = () => {
    clearTimeout(timeout);
    timeout = null;
  };

  return debounced;
};

export default debounce;
