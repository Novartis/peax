const camelToConst = str => str.split(/(?=[A-Z])/).join('_').toUpperCase();

export default camelToConst;
