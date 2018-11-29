import PropTypes from 'prop-types';
import React from 'react';

// Styles
import './ExampleList.scss';

const ExampleList = props => (
  <ul className="example-list flex-c flex-w-w no-list-style">
    {props.examples.map((example, index) => (
      <li className={example.columns === 2 && 'two-columns'} key={index}>
        <a href={example.url && example.url}>
          <figure
            className="example-figure"
            style={{
              backgroundImage: `url(${example.image ? example.image : ''})`
            }}
          />
        </a>
        {example.title && (
          <a href={example.url && example.url}>{example.title}</a>
        )}
        {example.description && (
          <p className="smaller">{example.description}</p>
        )}
        {example.location && (
          <div className="smaller one-line location">
            <input onClick="this.select();" value={example.location} />
          </div>
        )}
      </li>
    ))}
  </ul>
);

ExampleList.propTypes = {
  examples: PropTypes.array
};

export default ExampleList;
