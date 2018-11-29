import PropTypes from 'prop-types';
import React from 'react';

const withList = getKey => Component => {
  const List = ({ list }) => (
    <ul className="list no-list-style">
      {list.map(item => (
        <li className="list-item" key={getKey(item)}>
          <Component {...item} />
        </li>
      ))}
    </ul>
  );

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
