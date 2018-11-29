/* eslint prefer-template:0 */

/**
 * Get a cookie.
 *
 * @param {string} key - Cookie ID.
 * @return {string|null} If cookie exists, returns the content of the cookie.
 */
const get = key =>
  decodeURIComponent(
    document.cookie.replace(
      new RegExp(
        '(?:(?:^|.*;)\\s*' +
          encodeURIComponent(key).replace(/[-.+*]/g, '\\$&') +
          '\\s*\\=\\s*([^;]*).*$)|^.*$'
      ),
      '$1'
    )
  ) || null;

/**
 * Check if a cookie exists.
 *
 * @param {string} key - Cookie ID.
 * @return {boolean} If `true` a cookie with id `key` exists.
 */
const has = key =>
  new RegExp(
    '(?:^|;\\s*)' +
      encodeURIComponent(key).replace(/[-.+*]/g, '\\$&') +
      '\\s*\\='
  ).test(document.cookie);

/**
 * Remove a cookie.
 *
 * @param {string} key - Cookie ID.
 * @param {string} path - Cookie path.
 * @param {string} domain - Cookie domain.
 * @return {boolean} If `true` an existing cookie was removed, else `false`.
 */
const remove = (key, path, domain) => {
  if (!key || !has(key)) {
    return false;
  }
  document.cookie =
    encodeURIComponent(key) +
    '=; expires=Thu, 01 Jan 1970 00:00:00 GMT' +
    (domain ? '; domain=' + domain : '') +
    (path ? '; path=' + path : '');
  return true;
};

const set = (key, value, end, path, domain, secure) => {
  if (!key || /^(?:expires|max-age|path|domain|secure)$/i.test(key)) {
    return false;
  }

  let sExpires = '';

  if (end) {
    switch (end.constructor) {
      case Number:
        sExpires =
          end === Infinity
            ? '; expires=Fri, 31 Dec 9999 23:59:59 GMT'
            : '; max-age=' + end;
        break;
      case String:
        sExpires = '; expires=' + end;
        break;
      case Date:
        sExpires = '; expires=' + end.toUTCString();
        break;
      default:
      // Nothing
    }
  }
  document.cookie =
    encodeURIComponent(key) +
    '=' +
    encodeURIComponent(value) +
    sExpires +
    (domain ? '; domain=' + domain : '') +
    (path ? '; path=' + path : '') +
    (secure ? '; secure' : '');
  return true;
};

export default {
  get,
  has,
  remove,
  set
};
