const loadViewConfig = file =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.addEventListener('load', fileEvent => {
      try {
        resolve(JSON.parse(fileEvent.target.result));
      } catch (e) {
        reject(new Error('Only drop valid JSON'), e);
      }
    });

    try {
      reader.readAsText(file);
    } catch (e) {
      reject(new Error('Only drop actual files'));
    }
  });

export default loadViewConfig;
