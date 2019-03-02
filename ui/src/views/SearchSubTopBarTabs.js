import PropTypes from 'prop-types';
import React from 'react';
import { connect } from 'react-redux';

// Components
import Button from '../components/Button';
import SubTopBar from '../components/SubTopBar';
import TabTrigger from '../components/TabTrigger';

// Services
import { setSearchTab } from '../actions';

// Configs
import { TAB_CLASSIFICATIONS, TAB_RESULTS, TAB_SEEDS } from '../configs/search';

// Utils
import { Logger } from '../utils';

const logger = Logger('SearchSubTopBarTabs'); // eslint-disable-line

const SearchSubTopBarTabs = props => (
  <SubTopBar className="search-tab-triggers" stretch={true}>
    <TabTrigger
      for={TAB_SEEDS}
      tabChange={props.setTab}
      tabOpen={props.tab}
      className="rel flex-g-1"
    >
      <Button className="full-wh">New Samples</Button>
    </TabTrigger>
    <TabTrigger
      for={TAB_RESULTS}
      tabChange={props.setTab}
      tabOpen={props.tab}
      className="rel flex-g-1"
    >
      <Button className="full-wh">Results</Button>
    </TabTrigger>
    <TabTrigger
      for={TAB_CLASSIFICATIONS}
      tabChange={props.setTab}
      tabOpen={props.tab}
      className="rel flex-g-1"
    >
      <Button className="full-wh">Labels</Button>
    </TabTrigger>
  </SubTopBar>
);

SearchSubTopBarTabs.defaultProps = {
  minClassifications: Infinity,
  numClassifications: 0
};

SearchSubTopBarTabs.propTypes = {
  minClassifications: PropTypes.number,
  numClassifications: PropTypes.number,
  setTab: PropTypes.func.isRequired,
  tab: PropTypes.oneOfType([PropTypes.string, PropTypes.symbol]).isRequired
};

const mapStateToProps = state => ({
  tab: state.present.searchTab
});

const mapDispatchToProps = dispatch => ({
  setTab: searchTab => dispatch(setSearchTab(searchTab))
});

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(SearchSubTopBarTabs);
