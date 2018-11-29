const numToCassif = num => {
  switch (num) {
    case -1:
      return 'negative';
    case 1:
      return 'positive';
    case 0:
    default:
      return 'neutral';
  }
};

export default numToCassif;
