import downloadAsFile from './download-as-file';

const downloadAsJson = (filename, obj) => {
  downloadAsFile(
    filename,
    new Blob(
      [
        JSON.stringify(obj, null, 2),
      ],
      {
        type: 'application/json',
      }
    )
  );
};

export default downloadAsJson;
