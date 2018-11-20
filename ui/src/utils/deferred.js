const Deferred = function Deferred() {
  this.promise = new Promise((resolve, reject) => {
    this.resolve = resolve;
    this.reject = reject;
  });
};

Deferred.prototype.catch = function deferredCatch(callback) {
  this.promise.catch(callback);
  return this;
};

Deferred.prototype.finally = function deferredFinally(callback) {
  this.promise.then(() => {}).catch(() => {}).then(callback);
  return this;
};

Deferred.prototype.then = function deferredThen(callback) {
  this.promise.then(callback);
  return this;
};

export default Deferred;
