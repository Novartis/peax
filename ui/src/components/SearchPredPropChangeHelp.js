import React from 'react';

const SearchPredPropChangeHelp = () => (
  <div>
    <p>
      The prediction probability change plot shows the average change in the
      prediction probabilities of all regions in the dataset per trained
      classifiers.
    </p>
    <p>
      The change in prediction probabilities <code>&Delta;p</code> is defined as
      follows, where{' '}
      <code>
        p<sub>i</sub>
      </code>{' '}
      are the prediction probabilities of the <code>i</code>th classifier:
    </p>
    <pre>
      <code>
        &Delta;p = abs(p<sub>i</sub> - p<sub>i-1</sub>)
      </code>
    </pre>
    <p>
      A bar in the bar chart shows the average change in prediction
      probabilities among all regions in the dataset. The outer bar represents
      all regions and the inner bar represents all labeled regions.
    </p>
    <p>
      Typically, the change in the prediction probabilities is initially high as
      the first trained classifiers are not so stable. After some round of
      labeling regions and training classifiers the change in prediction
      probabilities should decrease to <code>0</code>.
    </p>
  </div>
);

export default SearchPredPropChangeHelp;
