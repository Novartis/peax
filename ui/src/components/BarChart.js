import { boundMethod } from 'autobind-decorator';
import { axisLeft, axisBottom } from 'd3-axis';
import { easeCubic } from 'd3-ease';
import { scaleLinear } from 'd3-scale';
import { transition } from 'd3-transition';
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
    this.margin = { top: 12, right: 0, bottom: 18, left: 32 };

    this.isChartInit = false;
    this.isChartPrepared = false;

    this.prepareData();
  }

  componentDidUpdate(prevProps) {
    let render = false;
    let prepareChart = false;
    if (prevProps.x !== this.props.x) {
      this.prepareData();
      prepareChart = true;
      render = true;
    }
    if (prevProps.parentWidth !== this.props.parentWidth) {
      this.getDimensions(this.svg.node());
      prepareChart = true;
      render = true;
    }
    if (prepareChart) {
      this.prepareChart();
      render = true;
    }
    if (render) {
      this.renderBarChart();
    }
  }

  prepareData() {
    this.xRel = this.props.x.reduce(
      (rel, v, i) => [...rel, v - (rel[i - 1] || 0)],
      []
    );
    this.hasY2 = this.props.diverging
      ? this.props.y2.length === this.props.y.length &&
        this.props.y4.length === this.props.y3.length
      : this.props.y2.length === this.props.y.length;
    this.data = this.props.x.map((x, i) => ({
      x,
      y: this.props.y[i],
      y2: this.props.y2[i],
      y3: this.props.y3[i],
      y4: this.props.y4[i],
      w: x
    }));
  }

  prepareChart() {
    const plotHeight = this.height - this.margin.top - this.margin.bottom;

    this.xScale = scaleLinear()
      .domain([0, max(this.props.x) + mean(this.xRel)])
      .range([this.margin.left, this.width - this.margin.right]);

    this.yScale = scaleLinear()
      .domain([this.props.yMin, this.props.yMax])
      .range([this.height - this.margin.bottom, this.margin.top]);

    this.yScaleTop = scaleLinear()
      .domain([this.props.yMin, this.props.yMax])
      .range([plotHeight / 2 + this.margin.top, this.margin.top]);

    this.yScaleBottom = scaleLinear()
      .domain([this.props.yMax, this.props.yMin])
      .range([
        this.height - this.margin.bottom,
        plotHeight / 2 + this.margin.top
      ]);

    this.xAxis = g =>
      g
        .attr('transform', `translate(0,${this.height - this.margin.bottom})`)
        .call(
          axisBottom(this.xScale)
            .ticks(5)
            .tickSizeOuter(0)
        );

    this.xAxisMiddle = g =>
      g
        .attr('transform', `translate(0,${plotHeight / 2 + this.margin.top})`)
        .call(
          axisBottom(this.xScale)
            .ticks(0)
            .tickSizeOuter(0)
        );

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

    this.yAxisTop = g =>
      g
        .attr('transform', `translate(${this.margin.left},0)`)
        .call(
          axisLeft(this.yScaleTop).tickValues([
            this.props.yMin,
            this.props.yMax / 2,
            this.props.yMax
          ])
        )
        .call(gg => gg.select('.domain').remove());

    this.yAxisBottom = g =>
      g
        .attr('transform', `translate(${this.margin.left},0)`)
        .call(
          axisLeft(this.yScaleBottom).tickValues([
            this.props.yMax,
            this.props.yMax / 2,
            this.props.yMin
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

    this.yGridlinesTop = g =>
      g
        .attr('transform', `translate(${this.margin.left},0)`)
        .call(
          axisLeft(this.yScaleTop)
            .tickFormat('')
            .tickValues([this.props.yMin, this.props.yMax / 2, this.props.yMax])
            .tickSize(-this.width)
        )
        .call(gg => gg.select('.domain').remove());

    this.yGridlinesBottom = g =>
      g
        .attr('transform', `translate(${this.margin.left},0)`)
        .call(
          axisLeft(this.yScaleBottom)
            .tickFormat('')
            .tickValues([this.props.yMin, this.props.yMax / 2, this.props.yMax])
            .tickSize(-this.width)
        )
        .call(gg => gg.select('.domain').remove());

    this.isChartPrepared = true;
  }

  initChart() {
    this.defs = this.svg.append('svg:defs');
    this.marker = this.defs.append('marker');
    this.marker.attr('id', 'arrow');
    this.marker.attr('class', 'x-axis-arrow');
    this.marker.attr('viewBox', '0 0 10 10');
    this.marker.attr('refX', '10');
    this.marker.attr('refY', '5');
    this.marker.attr('markerWidth', '6');
    this.marker.attr('markerHeight', '6');
    this.marker.attr('orient', 'auto-start-reverse');
    this.marker.append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z');

    this.yGridlinesG = this.svg.append('g').attr('class', 'y-gridlines');
    this.yGridlinesBottomG = this.svg
      .append('g')
      .attr('class', 'y-gridlines-bottom');
    this.baseBarsG = this.svg.append('g').attr('class', 'base-bars');
    this.primaryBarsG = this.svg.append('g').attr('class', 'primary-bars');
    this.baseBarsBottomG = this.svg
      .append('g')
      .attr('class', 'base-bars-bottom');
    this.primaryBarsBottomG = this.svg
      .append('g')
      .attr('class', 'primary-bars-bottom');
    this.xAxisG = this.svg.append('g').attr('class', 'x-axis');
    this.xAxisMiddleG = this.svg.append('g').attr('class', 'x-axis-middle');
    this.yAxisG = this.svg.append('g').attr('class', 'y-axis');
    this.yAxisBottomG = this.svg.append('g').attr('class', 'y-axis-bottom');

    this.isChartInit = true;
  }

  getDimensions(el) {
    const bBox = el.getBoundingClientRect();
    this.width = bBox.width;
    this.height = bBox.height;
  }

  @boundMethod
  onRef(svg) {
    if (!svg) return;

    this.getDimensions(svg);
    this.svg = select(svg);
    this.renderBarChart();
  }

  renderBarChart() {
    if (!this.isChartInit) this.initChart();
    if (!this.isChartPrepared) this.prepareChart();

    let yScale = this.yScale;

    if (this.props.diverging) {
      yScale = this.yScaleTop;
      this.yGridlinesG.call(this.yGridlinesTop);
      this.yGridlinesBottomG.call(this.yGridlinesBottom);
    } else {
      this.yGridlinesG.call(this.yGridlines);
    }

    const t = transition()
      .duration(750)
      .ease(easeCubic);

    let join = this.baseBarsG.selectAll('rect').data(this.data);

    join
      .enter()
      .append('rect')
      .attr('y', yScale(0))
      .attr('x', d => this.xScale(d.x) - this.props.barWidth / 2)
      .attr('width', this.props.barWidth)
      .transition(t)
      .attr('y', d => yScale(d.y))
      .attr('height', d => yScale(0) - yScale(d.y));

    join
      .transition(t)
      .attr('x', d => this.xScale(d.x) - this.props.barWidth / 2);

    if (this.props.diverging) {
      join = this.baseBarsBottomG.selectAll('rect').data(this.data);

      join
        .enter()
        .append('rect')
        .attr('x', d => this.xScale(d.x) - this.props.barWidth / 2)
        .attr('y', this.yScaleBottom(0))
        .attr('width', this.props.barWidth)
        .attr('height', 0)
        .transition(t)
        .attr('height', d => this.yScaleBottom(d.y3) - this.yScaleBottom(0));

      join
        .transition(t)
        .attr('x', d => this.xScale(d.x) - this.props.barWidth / 2);
    }

    if (this.hasY2) {
      join = this.primaryBarsG.selectAll('rect').data(this.data);

      join
        .enter()
        .append('rect')
        .attr('x', d => this.xScale(d.x) - this.props.barWidthSecondary / 2)
        .attr('y', yScale(0))
        .attr('width', this.props.barWidthSecondary)
        .attr('height', 0)
        .transition(t)
        .attr('y', d => yScale(d.y2))
        .attr('height', d => yScale(0) - yScale(d.y2) + 1);

      join
        .transition(t)
        .attr('x', d => this.xScale(d.x) - this.props.barWidthSecondary / 2);

      if (this.props.diverging) {
        join = this.primaryBarsBottomG.selectAll('rect').data(this.data);

        join
          .enter()
          .append('rect')
          .attr('x', d => this.xScale(d.x) - this.props.barWidthSecondary / 2)
          .attr('y', this.yScaleBottom(0))
          .attr('width', this.props.barWidthSecondary)
          .attr('height', 0)
          .transition(t)
          .attr('height', d => this.yScaleBottom(d.y4) - this.yScaleBottom(0));

        join
          .transition(t)
          .attr('x', d => this.xScale(d.x) - this.props.barWidthSecondary / 2);
      }
    }

    this.xAxisG.call(this.xAxis);
    this.xAxisG.select('path').attr('marker-end', 'url(#arrow)');

    if (this.props.diverging) {
      this.xAxisG.call(g => g.select('.domain').remove());
      this.xAxisMiddleG.call(this.xAxisMiddle);
      this.xAxisMiddleG.select('path').attr('marker-end', 'url(#arrow)');
      this.yAxisG.call(this.yAxisTop);
      this.yAxisBottomG.call(this.yAxisBottom);
    } else {
      this.yAxisG.call(this.yAxis);
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
  barWidth: 10,
  barWidthSecondary: 4,
  diverging: false,
  parentWidth: null,
  x: [],
  y: [],
  y2: [],
  y3: [],
  y4: [],
  yMax: 1,
  yMin: 0
};

BarChart.propTypes = {
  barWidth: PropTypes.number,
  barWidthSecondary: PropTypes.number,
  diverging: PropTypes.bool,
  parentWidth: PropTypes.number,
  x: PropTypes.array,
  y: PropTypes.array,
  y2: PropTypes.array,
  y3: PropTypes.array,
  y4: PropTypes.array,
  yMax: PropTypes.number,
  yMin: PropTypes.number
};

export default BarChart;
