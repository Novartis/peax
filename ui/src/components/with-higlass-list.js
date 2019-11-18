import { boundMethod } from 'autobind-decorator';
import PropTypes from 'prop-types';
import React from 'react';
import HiGlassViewer from './HiGlassViewer';

import { forwardEvent } from '../utils';

const getViewId = list =>
  list
    .reduce(
      (viewId, { searchId, windowId, showAutoencodings }) =>
        `${viewId}+${searchId}.${windowId}.${showAutoencodings ? 'e' : ''}`,
      ''
    )
    .slice(1);

let higlassList;
let higlassListWrapper;
let higlassScrollContainer;

const scrollHandler = () => {
  if (higlassScrollContainer) {
    higlassScrollContainer.scrollTop = higlassListWrapper.scrollTop;
  }
};

const forwardEventToHiGlass = event => {
  forwardEvent(event.nativeEvent, higlassScrollContainer);
};

const withList = getKey => Component => {
  class List extends React.Component {
    constructor(props) {
      super(props);

      this.viewConfigId = getViewId(this.props.list);
      this.state = {
        higlassApi: null,
        higlassLoaded: false
      };

      this.viewConfigId = getViewId(this.props.list);
    }

    componentDidUpdate() {
      const newViewConfigId = getViewId(this.props.list);

      if (this.viewConfigId !== newViewConfigId) {
        this.viewConfigId = newViewConfigId;
        // Unset scroll position
        higlassListWrapper.scrollTop = 0;
        higlassScrollContainer.scrollTop = 0;
      }

      this.checkScrollability();
    }

    checkScrollability() {
      if (higlassList) {
        if (
          higlassList.parentNode.getBoundingClientRect().height <
          higlassList.getBoundingClientRect().height
        ) {
          higlassList.style.marginRight = '0';
        } else {
          higlassList.style.marginRight = '0.5rem';
        }
      }
    }

    @boundMethod
    loadedHandler() {
      this.setState({
        higlassLoaded: true
      });
    }

    @boundMethod
    apiHandler(higlassApi) {
      higlassScrollContainer = higlassApi.getComponent().scrollContainer;
      this.setState({
        higlassApi
      });
    }

    @boundMethod
    higlassListRefHandler(element) {
      higlassList = element;
      this.checkScrollability();
    }

    render() {
      return (
        <div className="higlass-list-single-higlass-instance">
          <div
            ref={element => {
              higlassListWrapper = element;
            }}
            className="higlass-list-single-higlass-instance-list-wrapper"
            onScroll={scrollHandler}
            style={{ opacity: this.state.higlassLoaded ? 1 : 0 }}
          >
            <ul ref={this.higlassListRefHandler} className="list no-list-style">
              {this.props.list.map((item, index) => (
                <li className="list-item" key={getKey(item)}>
                  <Component
                    hgApi={this.state.higlassApi}
                    onMouseMove={forwardEventToHiGlass}
                    onMouseOut={forwardEventToHiGlass}
                    onMouseOver={forwardEventToHiGlass}
                    viewNumber={index}
                    {...item}
                  />
                </li>
              ))}
            </ul>
          </div>
          <div className="higlass-list-single-higlass-instance-higlass-wrapper">
            <HiGlassViewer
              api={this.apiHandler}
              containerPadding={0}
              isGlobalMousePosition
              isNotEditable
              isPixelPrecise
              isStatic
              isZoomFixed
              onLoaded={this.loadedHandler}
              sizeMode={'scroll'}
              viewConfigId={getViewId(this.props.list)}
              viewMargin={[8, 24, 0, 8]}
              viewPadding={0}
            />
          </div>
        </div>
      );
    }
  }

  List.propTypes = {
    list: PropTypes.arrayOf(
      PropTypes.shape({
        id: PropTypes.oneOfType([
          PropTypes.number,
          PropTypes.string,
          PropTypes.symbol
        ])
      })
    )
  };

  return List;
};

export default withList;
