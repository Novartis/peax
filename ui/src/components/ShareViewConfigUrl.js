import PropTypes from 'prop-types';
import React from 'react';

// Components
import ButtonIcon from './ButtonIcon';

// Styles
import './ShareViewConfigUrl.scss';

const ShareViewConfigUrl = props => (
  <div className='share-view-config-url'>
    <p>
      Copy the unique ID or URL below and share it with your colleagues,
      friends, fans, and beloved ones.
    </p>
    <label className='flex-c flex-a-c'>
      <strong className='share-view-config-url-label'>ID: </strong>
      <input
        type="text"
        value={props.id}
        className='flex-g-1'
        onFocus={e => e.target.select()}
      />
    </label>
    <label className='flex-c flex-a-c'>
      <strong className='share-view-config-url-label'>URL: </strong>
      <input
        type="url"
        value={props.url}
        className='flex-g-1 pr'
        onFocus={e => e.target.select()}
      />
      <ButtonIcon icon='external' href={props.url} external={true} />
    </label>
    <p><em>
      Note: there is currently no way to update a shared view. Just share it
      again under a new ID.
    </em></p>
  </div>
);

ShareViewConfigUrl.propTypes = {
  id: PropTypes.string.isRequired,
  url: PropTypes.string.isRequired,
};

export default ShareViewConfigUrl;
