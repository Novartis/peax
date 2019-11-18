import PropTypes from 'prop-types';
import React from 'react';

// Components
import Button from './Button';
import Spinner from './Spinner';

// Utils
import { range } from '../utils';

const getCurrentPage = (list, page = 0, itemsPerPage = 10) =>
  list.slice(page * itemsPerPage, (page + 1) * itemsPerPage);

const withPagination = () => Component => {
  const Pagination = ({
    list,
    isLoadingMore,
    isMoreLoadable,
    onLoadMore,
    onPage,
    textLoadMore,
    textNext,
    textPrev,
    page,
    pageTotal,
    itemsPerPage,
    ...otherProps
  }) => (
    <div className="with-pagination">
      <Component
        list={getCurrentPage(list, page, itemsPerPage)}
        {...otherProps}
      />

      <footer className="flex-c flex-a-c flex-jc-c with-pagination-actions">
        {page > 0 && (
          <Button
            className="with-pagination-next-prev"
            onClick={() => onPage(page - 1)}
            isDisabled={isLoadingMore}
          >
            {textPrev}
          </Button>
        )}
        {(page > 0 || pageTotal > page + 1) && (
          <ul className="flex-c no-list-style">
            {pageTotal && page > 2 && (
              <li>
                <Button
                  className="with-pagination-page-button"
                  isDisabled={isLoadingMore}
                  onClick={() => onPage(0)}
                >
                  1
                </Button>
              </li>
            )}
            {pageTotal && page > 3 && <li>&hellip;</li>}
            {range(Math.max(1, page - 1), page + 1).map(pageNum => (
              <li key={pageNum}>
                <Button
                  className="with-pagination-page-button"
                  isDisabled={isLoadingMore}
                  onClick={() => onPage(pageNum - 1)}
                >
                  {pageNum}
                </Button>
              </li>
            ))}
            <li key={page + 1}>
              <Button
                className="with-pagination-page-button-current"
                isDisabled={isLoadingMore}
              >
                {page + 1}
              </Button>
            </li>
            {pageTotal &&
              pageTotal > page + 2 &&
              range(page + 2, Math.min(pageTotal, page + 4)).map(pageNum => (
                <li key={pageNum}>
                  <Button
                    className="with-pagination-page-button"
                    isDisabled={isLoadingMore}
                    onClick={() => onPage(pageNum - 1)}
                  >
                    {pageNum}
                  </Button>
                </li>
              ))}
            {pageTotal && pageTotal > page + 4 && <li>&hellip;</li>}
            {pageTotal && pageTotal > page + 1 && (
              <li>
                <Button
                  className="with-pagination-page-button"
                  isDisabled={isLoadingMore}
                  onClick={() => onPage(pageTotal - 1)}
                >
                  {pageTotal}
                </Button>
              </li>
            )}
          </ul>
        )}
        {page !== pageTotal && (!pageTotal || pageTotal > page + 1) && (
          <Button
            className={`with-pagination-next-prev has-spinner ${
              isLoadingMore ? 'is-spinning' : ''
            }`}
            onClick={() => onPage(page + 1)}
            isDisabled={isLoadingMore}
          >
            <span className="text">{textNext}</span>
            {isLoadingMore && <Spinner height="1em" width="1em" />}
          </Button>
        )}
        {page + 1 === pageTotal && isMoreLoadable && (
          <Button
            className={`with-pagination-next-prev has-spinner ${
              isLoadingMore ? 'is-spinning' : ''
            }`}
            onClick={onLoadMore}
            isDisabled={isLoadingMore}
          >
            <span className="text">{textLoadMore}</span>
            {isLoadingMore && <Spinner height="1em" width="1em" />}
          </Button>
        )}
      </footer>
    </div>
  );

  Pagination.defaultProps = {
    isMoreLoadable: false,
    textLoadMore: 'Load More',
    textNext: 'Next',
    textPrev: 'Prev'
  };

  Pagination.propTypes = {
    page: PropTypes.number,
    pageTotal: PropTypes.number,
    list: PropTypes.array,
    isLoadingMore: PropTypes.bool,
    isMoreLoadable: PropTypes.bool,
    itemsPerPage: PropTypes.number,
    textLoadMore: PropTypes.string,
    textNext: PropTypes.string,
    textPrev: PropTypes.string,
    onLoadMore: PropTypes.func,
    onPage: PropTypes.func
  };

  return Pagination;
};

export default withPagination;
