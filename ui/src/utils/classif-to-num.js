const classifToNum = num => {
  switch (num[0]) {
    case 'n':
      return -1;
    case 'p':
      return 1;
    default:
      return 0;
  }
};

export default classifToNum;
