import { PropTypes } from 'prop-types';
import React from 'react';

// Components
import ButtonIcon from './ButtonIcon';

const AnnotationList = props => (
  <ol className={`annotation-list ${props.className}`}>
    {props.annotations.map(annotation => (
      <li
        className='flex-c'
        key={annotation.id}
      >
        <input
          onChange={() => this.selectAnnotation(annotation.id)}
          type='checkbox'
          checked={annotation.isShown}
        />
        <div
          className='flex-g-1 annotation-list-title'
          onClick={() => this.selectAnnotation(annotation.id)}
        >
          {annotation.title}
        </div>
        <ButtonIcon
          icon='edit'
          iconOnly={true}
          isActive={props.activeAnnotationId === annotation.id}
          onClick={() => props.setActiveAnnotation(annotation.id)} />
      </li>
    ))}
  </ol>
);

AnnotationList.defaultProps = {
  annotations: [],
  className: '',
};

AnnotationList.propTypes = {
  activeAnnotationId: PropTypes.string,
  annotations: PropTypes.array,
  className: PropTypes.string,
  selectAnnotation: PropTypes.func,
  setActiveAnnotation: PropTypes.func,
};

export default AnnotationList;
