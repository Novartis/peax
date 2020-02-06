import { ChromosomeInfo } from 'higlass';
import PropTypes from 'prop-types';
import React from 'react';
import { connect } from 'react-redux';
import { Link } from 'react-router-dom';

// Higher-order components
import { withPubSub } from '../hocs/pub-sub';

// Components
import ButtonLikeLink from '../components/ButtonLikeLink';
import Content from '../components/Content';
import ContentWrapper from '../components/ContentWrapper';
import Footer from '../components/Footer';
import HiGlassViewer from '../components/HiGlassViewer';
import Icon from '../components/Icon';
import InfoBar from '../components/InfoBar';
import HomeSubTopBar from './HomeSubTopBar';

// Actions
import {
  setHomeInfoBarClose,
  setHiglassMouseTool,
  setSearchTab
} from '../actions';

// Utils
import {
  api,
  debounce,
  Logger,
  readableDate,
  removeHiGlassEventListeners
} from '../utils';

// Configs
import { PAN_ZOOM, SELECT } from '../configs/mouse-tools';
import { TAB_SEEDS } from '../configs/search';

// Stylesheets
import './Home.scss';

const logger = Logger('Home');

class Home extends React.Component {
  constructor(props) {
    super(props);

    this.hiGlassEventListeners = {};
    this.pubSubs = [];

    this.state = {
      rangeSelection: [null, null],
      searches: []
    };

    this.checkChromInfo();

    this.keyDownHandlerBound = this.keyDownHandler.bind(this);
    this.rangeSelectionHandlerDb = debounce(
      this.rangeSelectionHandler.bind(this),
      250
    );
  }

  /* ----------------------- React Life Cycle Methods ----------------------- */

  componentDidMount() {
    this.pubSubs.push(
      this.props.pubSub.subscribe('keydown', this.keyDownHandlerBound)
    );
    this.getSearches();
  }

  componentWillUnmount() {
    this.pubSubs.forEach(subscription =>
      this.props.pubSub.unsubscribe(subscription)
    );
    this.pubSubs = [];
    removeHiGlassEventListeners(this.hiGlassEventListeners, this.hgApi);
    this.hiGlassEventListeners = {};
  }

  componentDidUpdate() {
    this.checkChromInfo();
  }

  /* ---------------------------- Custom Methods ---------------------------- */

  async getSearches() {
    this.setState({ isLoadingSearches: true, isErrorSearches: false });

    const searches = await api.getAllSearchInfos(3);
    const isErrorSearches = searches.status !== 200;

    this.setState({
      isLoadingSearches: false,
      isErrorSearches,
      searches: searches.body
    });
  }

  checkHgApi(newHgApi) {
    if (this.hgApi !== newHgApi) {
      removeHiGlassEventListeners(this.hiGlassEventListeners, this.hgApi);
      this.hiGlassEventListeners = {};

      this.hgApi = newHgApi;

      this.checkHgEvents();
      this.checkHgSetup();
    }
  }

  checkHgEvents() {
    if (!this.hgApi) return;

    if (
      this.props.mouseTool === SELECT &&
      !this.hiGlassEventListeners.rangeSelection
    ) {
      this.hiGlassEventListeners.rangeSelection = {
        name: 'rangeSelection',
        id: this.hgApi.on('rangeSelection', this.rangeSelectionHandlerDb)
      };
    }
  }

  async checkHgSetup() {
    if (!this.hgApi) return;

    this.searchInfo = await api.getInfo();
    this.hgApi.setRangeSelection1dSize(
      this.searchInfo.windowSizeMin,
      this.searchInfo.windowSizeMax
    );
    this.hgApi.setRangeSelectionToInt();
  }

  checkChromInfo() {
    if (!this.props.viewConfig) return;

    let newChromInfoUrl;
    try {
      newChromInfoUrl = this.props.viewConfig.views[0].tracks.top[1]
        .chromInfoPath;
    } finally {
      // Nothing
    }

    if (!newChromInfoUrl || newChromInfoUrl === this.chromInfoUrl) return;

    this.chromInfoUrl = newChromInfoUrl;

    this.chromInfo = new ChromosomeInfo(this.chromInfoUrl);
  }

  keyDownHandler(event) {
    if (event.keyCode === 83 && !event.ctrlKey && !event.metaKey) {
      // S
      event.preventDefault();
      this.props.setMouseTool(SELECT);
    }

    if (event.keyCode === 90 && !event.ctrlKey && !event.metaKey) {
      // Z
      event.preventDefault();
      this.props.setMouseTool(PAN_ZOOM);
    }
  }

  rangeSelectionHandler(rangeSelection) {
    if (!rangeSelection.dataRange[0]) {
      this.setState({ rangeSelection: [null, null] });
    } else {
      this.setState({ rangeSelection: rangeSelection.dataRange[0] });
    }
  }

  async search(rangeSelection) {
    const response = await api.newSearch(rangeSelection);
    logger.info(response);
    if (response.status === 200) {
      await this.props.setSearchToSeeds();
      this.props.history.push(`/search/${response.body.id}`);
    }
  }

  getHgViewId(showAes = this.props.showAutoencodings) {
    return `default${showAes ? '.e' : ''}`;
  }

  /* -------------------------------- Render -------------------------------- */

  render() {
    this.checkHgEvents();
    const hgViewId = this.getHgViewId();

    return (
      <ContentWrapper name="home">
        <InfoBar
          isClose={this.props.homeInfoBarClose}
          isClosable={true}
          onClose={() =>
            this.props.setHomeInfoBarClose(!this.props.homeInfoBarClose)
          }
          wrap={true}
        >
          <div className="flex-c">
            <p className="column-1-2 m-r-1 home-info-intro">
              Peax is a tool to visually search and explore peaks or other kind
              of patterns in 1D epigenomic tracks like DNase-seq and ChIP-seq.
            </p>
            <div className="column-1-2 m-l-1 flex-c flex-v home-info-actions">
              <ol className="no-list-style">
                {this.state.searches.map(s => (
                  <li key={s.id} className="flex-c flex-jc-sb">
                    <Link to={`/search/${s.id}`}>
                      {api.name || `Search #${s.id}`}
                    </Link>
                    <time dateTime={s.updated}>{readableDate(s.updated)}</time>
                  </li>
                ))}
                {!this.state.searches.length && (
                  <li>
                    <em>No previsous searches found. So sad!</em>
                  </li>
                )}
              </ol>
              <ButtonLikeLink className="flex-g-1" to="/search">
                <div className="flex-c flex-a-c full-h">
                  <div className="flex-g-1">All searches</div>
                  <Icon iconId="arrow-right" />
                </div>
              </ButtonLikeLink>
            </div>
          </div>
        </InfoBar>
        <HomeSubTopBar
          rangeSelection={this.state.rangeSelection}
          search={this.search.bind(this)}
        />
        <Content name="home" rel={true} wrap={true}>
          <HiGlassViewer
            api={this.checkHgApi.bind(this)}
            enableAltMouseTools={true}
            viewConfigId={hgViewId}
          />
        </Content>
        <Footer />
      </ContentWrapper>
    );
  }
}

Home.propTypes = {
  history: PropTypes.object.isRequired,
  homeInfoBarClose: PropTypes.bool,
  mouseTool: PropTypes.string,
  pubSub: PropTypes.object.isRequired,
  setHomeInfoBarClose: PropTypes.func,
  setMouseTool: PropTypes.func,
  setSearchToSeeds: PropTypes.func,
  showAutoencodings: PropTypes.bool,
  viewConfig: PropTypes.object
};

const mapStateToProps = state => ({
  homeInfoBarClose: state.present.homeInfoBarClose,
  mouseTool: state.present.higlassMouseTool,
  showAutoencodings: state.present.showAutoencodings,
  viewConfig: state.present.viewConfig
});

const mapDispatchToProps = dispatch => ({
  setHomeInfoBarClose: isClosed => dispatch(setHomeInfoBarClose(isClosed)),
  setMouseTool: mouseTool => dispatch(setHiglassMouseTool(mouseTool)),
  setSearchToSeeds: () => dispatch(setSearchTab(TAB_SEEDS))
});

export default connect(mapStateToProps, mapDispatchToProps)(withPubSub(Home));
