import { boundMethod } from 'autobind-decorator';
import { axisLeft, axisBottom } from 'd3-axis';
import { scaleLinear } from 'd3-scale';
import { select } from 'd3-selection';
import { PropTypes } from 'prop-types';
import React from 'react';

import { max, mean } from '../utils';

// Styles
import './BarChart.scss';

class BarChart extends React.Component {
  constructor(props) {
    super(props);

    this.width = 240;
    this.height = 48;
    this.margin = { top: 12, right: 0, bottom: 16, left: 24 };

    this.isChartInit = false;
    this.isChartPrepared = false;

    this.prepareData();
  }

  componentDidUpdate(prevProps) {
    let prepareChart = false;
    if (prevProps.x !== this.props.x) {
      this.prepareData();
      prepareChart = true;
    }
    if (prevProps.parentWidth !== this.props.parentWidth) {
      this.getDimensions(this.svg.node());
      prepareChart = true;
    }
    if (prepareChart) {
      this.prepareChart();
    }
    this.renderBarChart();
  }

  getSize = y => y ** 2 * 8 + 4;

  getPath = d => {
    const w = this.getSize(d.y);
    const w2 = this.getSize(d.y2);
    const x0 = this.xScale(d.x) - (d.y2 > d.y ? w2 : w);
    const x1 = this.xScale(d.x) + (d.y2 > d.y ? w2 : w);
    const x2 = this.xScale(d.x) + (d.y2 < d.y ? w2 : w);
    const x3 = this.xScale(d.x) - (d.y2 < d.y ? w2 : w);
    const y01 = this.yScale(Math.max(d.y, d.y2));
    const y23 = this.yScale(Math.min(d.y, d.y2));

    return `M ${x0} ${y01} L ${x1} ${y01} L ${x2} ${y23} L ${x3} ${y23} Z`;
  };

  prepareData() {
    this.xRel = this.props.x.reduce(
      (rel, v, i) => [...rel, v - (rel[i - 1] || 0)],
      []
    );
    this.hasY2 = this.props.y2.length === this.props.y.length;
    this.data = this.props.x.map((x, i) => ({
      x,
      y: this.props.y[i],
      y2: this.props.y2[i],
      w: x
    }));
  }

  prepareChart() {
    this.xScale = scaleLinear()
      .domain([0, max(this.props.x) + mean(this.xRel)])
      .range([this.margin.left, this.width - this.margin.right]);

    this.yScale = scaleLinear()
      .domain([this.props.yMin, this.props.yMax])
      .range([this.height - this.margin.bottom, this.margin.top]);

    this.xAxis = g =>
      g
        .attr('transform', `translate(0,${this.height - this.margin.bottom})`)
        .call(axisBottom(this.xScale).tickSizeOuter(0));

    this.yAxis = g =>
      g
        .attr('transform', `translate(${this.margin.left},0)`)
        .call(
          axisLeft(this.yScale).tickValues([
            this.props.yMin,
            this.props.yMax / 2,
            this.props.yMax
          ])
        )
        .call(gg => gg.select('.domain').remove());

    this.yGridlines = g =>
      g
        .attr('transform', `translate(${this.margin.left},0)`)
        .call(
          axisLeft(this.yScale)
            .tickFormat('')
            .tickValues([
              this.props.yMin,
              this.props.yMax / 4,
              this.props.yMax / 2,
              (this.props.yMax * 3) / 4,
              this.props.yMax
            ])
            .tickSize(-this.width)
        )
        .call(gg => gg.select('.domain').remove());

    this.isChartPrepared = true;
  }

  initChart() {
    this.yGridlinesG = this.svg.append('g').attr('class', 'y-gridlines');
    this.connectingLinesG = this.svg
      .append('g')
      .attr('class', 'connecting-lines');
    this.baseCirclesG = this.svg.append('g').attr('class', 'base-circles');
    this.primaryCirclesG = this.svg
      .append('g')
      .attr('class', 'primary-circles');
    this.primaryCirclesDotG = this.svg
      .append('g')
      .attr('class', 'primary-circles-dot');
    this.xAxisG = this.svg.append('g').attr('class', 'x-axis');
    this.yAxisG = this.svg.append('g').attr('class', 'y-axis');

    this.isChartInit = true;
  }

  getDimensions(el) {
    const bBox = el.getBoundingClientRect();
    this.width = bBox.width;
    this.height = bBox.height;
  }

  @boundMethod
  onRef(svg) {
    this.getDimensions(svg);
    this.svg = select(svg);
    this.renderBarChart();
  }

  renderBarChart() {
    if (!this.isChartInit) this.initChart();
    if (!this.isChartPrepared) this.prepareChart();

    this.yGridlinesG.call(this.yGridlines);
    this.xAxisG.call(this.xAxis);
    this.yAxisG.call(this.yAxis);

    if (this.hasY2) {
      this.connectingLinesG
        .selectAll('path')
        .data(this.data)
        .join('path')
        .attr('d', this.getPath)
        .attr('fill', d => (d.y > d.y2 ? '#999' : 'steelblue'));
    }

    this.baseCirclesG
      .selectAll('circle')
      .data(this.data)
      .join('circle')
      .attr('cx', d => this.xScale(d.x))
      .attr('cy', d => this.yScale(d.y))
      .attr('r', d => this.getSize(d.y, true));

    if (this.hasY2) {
      this.primaryCirclesG
        .selectAll('circle')
        .data(this.data)
        .join('circle')
        .attr('cx', d => this.xScale(d.x))
        .attr('cy', d => this.yScale(d.y2))
        .attr('r', d => this.getSize(d.y2));

      this.primaryCirclesDotG
        .selectAll('circle')
        .data(this.data)
        .join('circle')
        .attr('cx', d => this.xScale(d.x))
        .attr('cy', d => this.yScale(d.y2))
        .attr('r', 1.5);
    }
  }

  render() {
    const classNames = 'bar-chart';

    return (
      <div className={classNames}>
        <svg ref={this.onRef} />
      </div>
    );
  }
}

BarChart.defaultProps = {
  parentWidth: null,
  x: [],
  y: [],
  y2: [],
  yMax: 1,
  yMin: 0
};

BarChart.propTypes = {
  parentWidth: PropTypes.number,
  x: PropTypes.array,
  y: PropTypes.array,
  y2: PropTypes.array,
  yMax: PropTypes.number,
  yMin: PropTypes.number
};

export default BarChart;
