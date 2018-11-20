import { PropTypes } from 'prop-types';
import React from 'react';

// Components
import Button from './Button';
import ButtonIcon from './ButtonIcon';
import Icon from './Icon';
import ToolTip from './ToolTip';

const AnnotationDetails = props => (
  <div className={`annotation-details ${props.className}`}>
    <div className='flex-c annotation-details-header'>
      <div className='flex-c flex-g-1'>
        <ToolTip
          delayIn={1000}
          delayOut={500}
          title='Last edited'
        >
          <Icon iconId='calendar'/>
          <div>3/4/2017</div>
        </ToolTip>
      </div>
      <ToolTip
        align='right'
        delayIn={1000}
        delayOut={500}
        title='Share'
      >
        <ButtonIcon icon='share' iconOnly={true} />
      </ToolTip>
    </div>
    <div className='annotation-details-form'>
      <input type='text' placeholder='Title' />
      <textarea placeholder='Title' />
    </div>
    <div className='annotation-details-loci'>
      <ol className='flex-c'>
        <div className="flex-g-1 flex-c flex-v"></div>
        <ToolTip
          align='right'
          delayIn={1000}
          delayOut={500}
          title='Remove'
        >
          <ButtonIcon icon='cross' iconOnly={true} />
        </ToolTip>
      </ol>
      <Button>Add</Button>
    </div>
  </div>
);

AnnotationDetails.defaultProps = {
  className: '',
};

AnnotationDetails.propTypes = {
  annotation: PropTypes.object,
  className: PropTypes.string,
  selectAnnotation: PropTypes.func,
  setActiveAnnotation: PropTypes.func,
};

export default AnnotationDetails;
