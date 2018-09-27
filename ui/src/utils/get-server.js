import { DEFAULT_SERVER_PORT } from '../configs';

const hostname = window.HGAC_SERVER || window.location.hostname;
const port = window.HGAC_SERVER_PORT || DEFAULT_SERVER_PORT;

const getServer = () => `//${hostname}:${port}`;

export default getServer;
