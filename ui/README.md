# HiGlass App

> The web application for viewing, exploring, and annotating 1D and 2D genomic data.

[![HiGlass](https://img.shields.io/badge/higlass-👍-red.svg?colorB=0f5d92)](http://higlass.io)
[![Build Status](https://img.shields.io/travis/hms-dbmi/higlass-app/master.svg?colorB=0f5d92)](https://travis-ci.org/hms-dbmi/higlass-app)

**URL**: http://higlass.io

**Note**: This is the source code for the web application only! You might want to check out the following repositories as well:

- HiGlass viewer: https://github.com/hms-dbmi/higlass
- HiGlass server: https://github.com/hms-dbmi/higlass-server
- HiGlass docker: https://github.com/hms-dbmi/higlass-docker

## Development

### Installation

```bash
$ git clone --recursive https://github.com/hms-dbmi/higlass-app && higlass-app
$ npm install
```

**Note**: If you forgot to add `--recursive` do the following to pull the submodules

```
$ git submodule update --init --recursive --remote
```

In order to update the wiki run:

```
$ git submodule update --recursive --remote
```


### Commands

**Developmental server**: `npm start`

**Production build**: `npm run build`

**Run tests**: `npm test`

**Update wiki**: `npm run update-wiki`


### Configuration

HiGlass App can be configured at build time using [`config.json`](config.json).
[`config.json`](config.json) contains the default configuration. To adjust it copy it to `config.dev.json`, `config.prod.json`, or `config.local.json`. `dev` is used in developmental mode, `prod` is picked up in production, and `local` can be used for local testing. The config is overwritten in the following order: _default_, dev, prod, local.


### Folder Structure

HiGlass App is based on [react-create-app](https://github.com/facebookincubator/create-react-app) and implements the following folder structure:

- **[`/build`](build)** [_Do not edit._]

  Contains the build files.

- **[`/config`](config)**

  Contains the build configs for webpack and converting Markdown into JSX. Except the JSX, the files are derived from [react-create-app](https://github.com/facebookincubator/create-react-app).

- **[`/node_modules`](node_modules)** [_Do not edit._]

  Contains thrid party libraries.

- **[`/public`](public)**

  Contains the public _index_ files of HiGlassApp.

- **[`/scripts`](scripts)**

  Contains node scripts for webpack and converting Markdown into JSX. Except the JSX, the files are derived from [react-create-app](https://github.com/facebookincubator/create-react-app).

- **[`/src`](src)**

  Entry point for the application source.

  - **[`/actions`](actions)**

    Flat folder containing Redux actions. All actions should be exported in the [`index.js`](src/actions/index.js).

  - **[`/components`](components)**

    Flat folder containing _presentational_ components. See [Dan's article](https://medium.com/@dan_abramov/smart-and-dumb-components-7ca2f9a7c7d0) for a comparison between presentational and container components.

  - **[`/configs`](configs)**

    Flat folder containing config files. Configs should host all constant variables.

  - **[`/containers`](containers)**

    Flat folder containing _container_ components. See [Dan's article](https://medium.com/@dan_abramov/smart-and-dumb-components-7ca2f9a7c7d0) for a comparison between presentational and container components.

  - **[`/images`](images)**

    Flat folder containing images.

  - **[`/reducers`](reducers)**

    Flat folder containing Redux reducers.

  - **[`/services`](services)**

    Flat folder containing services. (A service is considered to be an object or function with a app-wide state, not managed by the Redux store. E.g., the Redux store itself is a service to provide convinient helper functions for _undo_ etc.)

  - **[`/styles`](styles)**

    Flat folder containing generic style components. Note: this is kind of arbitrary but I like to separate reusable CSS classes from [`index.scss`](src/index.scss) such that I can quickly drop them into other apps.

  - **[`/utils`](utils)**

    Flat folder containing utility functions. (Utility function **must** be pure and should not have any state.)

  - **[`/views`](views)**

    Flat folder containing all view components. (View components are just like presentational components with the only difference that they are associated to a specific URL. This separation is again arbitrary but I like having all views separated as I consider them **uncomposable**. Only [`Main`](src/components/Main.js) should import views.)

