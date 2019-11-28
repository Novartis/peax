import React from 'react';

const SearchUncertaintyHelp = () => (
  <div>
    <p>
      The uncertainty plot shows the overall uncertainty of the trained
      classifiers. Hereby, uncertainty is defined as follow, where{' '}
      <code>p</code> stands for the prediction probability:
    </p>
    <pre>
      <code>uncertainty = 1 - abs(p - 0.5) * 2</code>
    </pre>
    <p>
      In otherwords, the uncertainty is <code>1</code> when the prediction
      probability is <code>0.5</code> and the uncertainty is <code>0</code> when
      the prediction probability is either <code>0</code> or <code>1</code>.
    </p>
    <p>
      A bar in the bar chart shows the average uncertainty among all regions in
      the dataset. The outer bar represents all regions and the inner bar
      represents all labeled regions.
    </p>
    <p>
      Typically, the uncertainty of the classifier initially increases and
      should then decrease. A perfect classifier has an uncertainty of{' '}
      <code>0</code>.
    </p>
  </div>
);

export default SearchUncertaintyHelp;
