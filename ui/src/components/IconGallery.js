import React from 'react';

// Components
import Icon from './Icon';

// Config
import icons from '../configs/icons';

// Styles
import './IconGallery.scss';

const convertId = id => (id ? id.replace(/_/g, '-').toLowerCase() : '');

const IconGallery = () => (
  <ul className='icon-gallery no-list-style flex-c flex-w-w'>
    {Object.keys(icons).map(icon => (
      <li className='icon-gallery-tile flex-c flex-v flex-a-c' key={convertId(icon)}>
        <Icon iconId={convertId(icon)} className='icon-gallery-icon'/>
        <span className='icon-gallery-icon-id'>{convertId(icon)}</span>
      </li>
    ))}
  </ul>
);

export default IconGallery;
