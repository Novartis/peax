import update from 'immutability-helper';
import { decode } from 'tab64';

// Utils
import { getServer, mergeError, mergeJsonResponse } from '../utils';

// Non-persistent state
const state = {
  windowSizeMin: 0,
  windowSizeMax: Infinity,
  target: null,
  classification: [],
};

const server = `${getServer()}/api/v1`;

const info = fetch(`${server}/info/`)
  .then(mergeJsonResponse)
  .then((response) => {
    if (response.status !== 200) return false;
    return response.body;
  });

const getInfo = async () => info;

const newSearch = async rangeSelection => fetch(
  `${server}/search/`, {
    method: 'post',
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
    body: JSON.stringify({ window: rangeSelection }),
  }
).then(mergeJsonResponse).catch(mergeError);

const getSearchInfo = async searchId => fetch(`${server}/search/?id=${searchId}`)
  .then(mergeJsonResponse).catch(mergeError);

const getAllSearchInfos = async (max = 0) => fetch(`${server}/search/?max=${max}`)
  .then(mergeJsonResponse).catch(mergeError);

const setClassification = async (searchId, windowId, classification) => fetch(
  `${server}/classification/`, {
    method: 'put',
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
    body: JSON.stringify({ searchId, windowId, classification }),
  }
).then(mergeJsonResponse).catch(mergeError);

const deleteClassification = async (searchId, windowId) => fetch(
  `${server}/classification/`, {
    method: 'delete',
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
    body: JSON.stringify({ searchId, windowId }),
  }
).then(mergeJsonResponse).catch(mergeError);

const getClassifications = async searchId => fetch(`${server}/classifications/?s=${searchId}`)
  .then(mergeJsonResponse).catch(mergeError);

const getDataTracks = async () => fetch(`${server}/data-tracks/`)
  .then(mergeJsonResponse).catch(mergeError);

const getPredictions = async searchId => fetch(`${server}/predictions/?s=${searchId}`)
  .then(mergeJsonResponse).catch(mergeError);

const getSeeds = async searchId => fetch(`${server}/seeds/?s=${searchId}`)
  .then(mergeJsonResponse).catch(mergeError);

const newClassifier = async searchId => fetch(
  `${server}/classifier/?s=${searchId}`, { method: 'post' }
).then(mergeJsonResponse).catch(mergeError);

const getClassifier = async searchId => fetch(`${server}/classifier/?s=${searchId}`)
  .then(mergeJsonResponse).catch(mergeError);

const newProjection = async searchId => fetch(
  `${server}/projection/?s=${searchId}`, { method: 'put' }
).then(mergeJsonResponse).catch(mergeError);

const getProjection = async searchId => fetch(`${server}/projection/?s=${searchId}`)
  .then(mergeJsonResponse)
  .then(resp => update(
    resp,
    {
      body: (body = {}) => update(
        body,
        { projection: { $set: decode(body.projection || '', 'float32') } }
      ),
    }
  ))
  .catch(mergeError);

const getProbabilities = async searchId => fetch(`${server}/probabilities/?s=${searchId}`)
  .then(mergeJsonResponse)
  .then(resp => update(
    resp,
    {
      body: (body = {}) => update(
        body,
        { results: { $set: decode(body.results || '', 'float32') } }
      ),
    }
  ))
  .catch(mergeError);

const getClasses = async searchId => fetch(`${server}/classes/?s=${searchId}`)
  .then(mergeJsonResponse)
  .then(resp => update(
    resp,
    {
      body: (body = {}) => update(
        body,
        { results: { $set: decode(body.results || '', 'uint8') } }
      ),
    }
  ))
  .catch(mergeError);

const search = {
  deleteClassification,
  getAllSearchInfos,
  getClasses,
  getClassifications,
  getClassifier,
  getDataTracks,
  getInfo,
  getPredictions,
  getProbabilities,
  getProjection,
  getSearchInfo,
  getSeeds,
  newClassifier,
  newProjection,
  newSearch,
  setClassification,
  state,
};

export default search;
