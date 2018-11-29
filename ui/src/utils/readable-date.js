const readableDate = (date, hasHour) => {
  let d = date;
  try {
    d.getMonth();
  } catch (e) {
    d = new Date(d);
  }

  const format = {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  };

  if (hasHour) {
    format.hour = 'numeric';
    format.minute = 'numeric';
  }

  return d.toLocaleString('en-us', format);
};

export default readableDate;
