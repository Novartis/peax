import React from 'react';

const SearchDivergenceHelp = () => (
  <div>
    <p>
      The convergence/divergence plot shows for how many regions the prediction
      probability converges and diverges per trained classifier.
    </p>
    <p>
      The convergence and divergence is defined by the change in the prediction
      probabilities over the last 3 classifiers. In the following{' '}
      <code>
        p<sub>i</sub>
      </code>{' '}
      stands for the prediction probabilities of the <code>i</code>the
      classifier:
    </p>
    <pre>
      <code>
        convergence = sign(p<sub>i-2</sub> - p<sub>i-1</sub>) == sign(p
        <sub>i-1</sub> - p<sub>i</sub>)
      </code>
      <br />
      <code>
        divergence &nbsp;= sign(p<sub>i-2</sub> - p<sub>i-1</sub>) != sign(p
        <sub>i-1</sub> - p<sub>i</sub>)
      </code>
    </pre>
    <p>
      A bar in the bar chart shows the percentage of regions in the dataset that
      converge or diverge. Note, that the two bars do not necessarily need to
      add up to 1 as there can be regions for which the prediction probability
      does not change at all. The outer bar represents all regions and the inner
      bar represents all labeled regions.
    </p>
    <p>
      Typically, the change in the convergence is initially higher and then
      gradually decreases. A bar close to <code>0</code> indicates that the
      classifier didn&apos;t change much. A large divergence can indicate that
      the recently added labels introduce a shift in the desired target pattern.
      For example, image you look for peaks and initially label sharp and broad
      peaks as positive but decide to exclude broad peaks after a few rounds of
      labeling regions and training classifiers. This will likely cause a big
      change in the prediction probabilities for several regions.
    </p>
  </div>
);

export default SearchDivergenceHelp;
