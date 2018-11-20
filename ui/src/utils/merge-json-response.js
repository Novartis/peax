const mergeJsonResponse = response =>
  response.json().then(body => ({ body, status: response.status }));

export default mergeJsonResponse;
