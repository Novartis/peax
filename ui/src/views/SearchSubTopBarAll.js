import PropTypes from 'prop-types';
import React from 'react';
import { connect } from 'react-redux';

// Components
import ButtonIcon from '../components/ButtonIcon';
import InputIcon from '../components/InputIcon';
import SubTopBar from '../components/SubTopBar';
import SubTopBottomBarButtons from '../components/SubTopBottomBarButtons';
import ToolTip from '../components/ToolTip';

// Utils
import { Logger } from '../utils';

const logger = Logger('SearchSubTopBarAll'); // eslint-disable-line

const SearchSubTopBarAll = props => (
  <SubTopBar>
    <SubTopBottomBarButtons className="flex-c flex-a-c no-list-style">
      <li>
        <ToolTip
          align="left"
          delayIn={1000}
          delayOut={500}
          title={
            <span className="flex-c">
              <span>Reset viewports</span>
              <span className="short-cut">R</span>
            </span>
          }
        >
          <ButtonIcon
            icon="reset"
            iconOnly={true}
            isDisabled={!props.viewportChanged}
            onClick={props.resetViewport()}
          />
        </ToolTip>
      </li>
    </SubTopBottomBarButtons>
    <SubTopBottomBarButtons className="flex-c flex-a-c flex-jc-e no-list-style">
      <li>
        <InputIcon icon="magnifier" placeholder="Search" />
      </li>
    </SubTopBottomBarButtons>
  </SubTopBar>
);

SearchSubTopBarAll.defaultProps = {
  viewportChanged: false
};

SearchSubTopBarAll.propTypes = {
  viewportChanged: PropTypes.bool,
  resetViewport: PropTypes.func.isRequired
};

const mapStateToProps = (/* state */) => ({});
const mapDispatchToProps = (/* dispatch */) => ({});

export default connect(mapStateToProps, mapDispatchToProps)(SearchSubTopBarAll);
