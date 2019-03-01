import React from 'react';

// Components
import Icon from '../components/Icon';

const SearchRightBarHelp = () => (
  <div className="right-bar-help flex-c flex-v full-wh">
    <ul className="search-right-bar-padding no-list-style compact-list compact-list-with-padding">
      <li>
        <strong>Hit: </strong>
        <span>
          A hit is a genomic window that is classified positive by the random
          forest model based on your labels and your probability threshold.
        </span>
      </li>
      <li>
        <strong>Label: </strong>
        <span>
          A label is the manual assignment of either a positive (
          <Icon iconId="checkmark" inline />) or negative (
          <Icon iconId="cross" inline />) class to a genomic window.
        </span>
      </li>
      <li>
        <strong>Training: </strong>
        <span>
          After labeling some genomic windows, a random forest classifier will
          be train.
        </span>
      </li>
      <li>
        <strong>Uncertainty: </strong>
        <span>
          Class prediction uncertainty is defined as the mean distance of
          classification probabilities from 0.5. An uncertainty of 1 is the
          worst and means that the classifier predicts the class for window with
          probability 0.5. An uncertainty of 0 is the best and means that the
          classifier predicts the class for window with either 1.0 or 0.0.
        </span>
      </li>
      <li>
        <strong>Variance: </strong>
        <span>
          Class prediction variance is defined as the mean variance of class
          predictions based on confidence intervals as proposed by{' '}
          <a
            href="http://jmlr.org/papers/v15/wager14a.html"
            target="_blank"
            rel="noopener noreferrer"
          >
            Wager et al. 2014
          </a>
        </span>
      </li>
    </ul>
  </div>
);

export default SearchRightBarHelp;
