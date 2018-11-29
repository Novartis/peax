const downloadAsFile = (filename, blob) => {
  const element = document.createElement('a');

  element.href = window.URL.createObjectURL(blob);
  element.download = filename;

  if (document.createEvent) {
    const event = document.createEvent('MouseEvents');
    event.initEvent('click', true, true);
    element.dispatchEvent(event);
  } else {
    element.click();
  }
};

export default downloadAsFile;
