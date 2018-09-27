const multiStorage = function multiStorage(storages, prefix) {
  if (storages.length !== 2) {
    throw Error('Currently 2 storages need to be provided.');
  }

  const safeStorages = storages.map((storage) => {
    if (storage.keys && !storage.getAllKeys) {
      // Fallback for localForage
      return { ...storage, getAllKeys: storage.keys };
    }

    return storage;
  });

  const primStorage = safeStorages[0];
  const secoStorage = safeStorages[1];

  const api = {
    getAllKeys(cb) {
      return primStorage.getAllKeys(cb);
    },
    getItem(key, cb) {
      return primStorage.getItem(key, cb);
    },
    setItem(key, string, cb) {
      secoStorage.setItem(key, string, cb);

      return primStorage.setItem(key, string, cb);
    },
    removeItem(key, cb) {
      secoStorage.removeItem(key, cb);

      return primStorage.removeItem(key, cb);
    },
  };

  return secoStorage.getAllKeys()
    .then(keys => keys.filter(key => key.indexOf(prefix) === 0))
    .then(secoKeys => primStorage.getAllKeys().then(keys => [
      keys, secoKeys,
    ]))
    .then((keys) => {
      const idx = keys[0].indexOf(`${prefix}index`);

      if (idx === -1) {
        // Reset primary from secondary storage.
        const reset = [];

        keys[1].forEach((key) => {
          reset.push(
            secoStorage.getItem(key)
              .then(value => primStorage.setItem(key, value))
          );
        });

        return Promise.all(reset).then(() => api);
      }

      return Promise.resolve(api);
    });
};

export default multiStorage;
