import canvasCamera2d from "canvas-camera-2d"; // eslint-disable-line import/no-unresolved
import { vec3 } from "gl-matrix";
import createMousePos from "mouse-position";
import createMousePrs from "mouse-pressed";
import createPubSub from "pub-sub-es";
import createRegl from "regl";
import createScroll from "scroll-speed";

import FRAG_SHADER from "./Scatterplot.fshader";
import VERT_SHADER from "./Scatterplot.vshader";

import { withRaf } from "../utils";

const DEFAULT_POINT_SIZE = 3;
const DEFAULT_POINT_SIZE_HIGHLIGHT = 3;
const DEFAULT_WIDTH = 100;
const DEFAULT_HEIGHT = 100;
const DEFAULT_PADDING = 0;
const DEFAULT_COLORMAP = [];
const DEFAULT_TARGET = [0, 0];
const DEFAULT_DISTANCE = 1;
const CLICK_DELAY = 250;

const dist = (x1, x2, y1, y2) => Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);

const Scatterplot = ({
  canvas = document.createElement("canvas"),
  colorMap = DEFAULT_COLORMAP,
  pointSize = DEFAULT_POINT_SIZE,
  pointSizeHighlight = DEFAULT_POINT_SIZE_HIGHLIGHT,
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
  padding = DEFAULT_PADDING
} = {}) => {
  const pubSub = createPubSub();
  let _canvas = canvas;
  let _width = width;
  let _height = height;
  let _padding = padding;
  let _pointSize = pointSize;
  let _pointSizeHighlight = pointSizeHighlight;
  let _colorMap = colorMap;
  let camera;
  let regl;
  let scroll;
  let mousePosition;
  let mousePressed;
  let mouseDown;
  let mouseDownTime;
  let mouseDownX;
  let mouseDownY;
  let _points = [];
  let _numHighlight = [];

  _canvas.width = _width * window.devicePixelRatio;
  _canvas.height = _height * window.devicePixelRatio;

  const raycast = () => {
    // Get the mouse cursor position
    mousePosition.flush();
    let { 0: x, 1: y } = mousePosition;
    // Get relative webgl coordinates
    x = -1 + (x / _width) * 2;
    y = 1 + (y / _height) * -2;
    // Normalize by the camera
    const v = [x, y, 1];
    vec3.transformMat3(v, v, camera.transformation);
    [x, y] = v;
    // Find the closest point
    let minDist = Infinity;
    let clostestPoint;
    _points.forEach(([ptX, ptY], i) => {
      const d = dist(ptX, x, ptY, y);
      if (d < minDist) {
        minDist = d;
        clostestPoint = i;
      }
    });
    if (minDist < (_pointSize / _width) * 2) return clostestPoint;
  };

  const mouseDownHandler = () => {
    mouseDown = true;
    mouseDownTime = performance.now();

    // Get the mouse cursor position
    mousePosition.flush();

    // Get relative webgl coordinates
    const { 0: x, 1: y } = mousePosition;
    mouseDownX = -1 + (x / _width) * 2;
    mouseDownY = 1 + (y / _height) * -2;
  };

  const mouseUpHandler = () => {
    mouseDown = false;
    if (performance.now() - mouseDownTime <= CLICK_DELAY) {
      pubSub.publish("click", {
        selectedPoint: raycast(mouseDownX, mouseDownY)
      });
    }
  };

  const initRegl = (c = _canvas) => {
    regl = createRegl({ canvas: c, extensions: ["OES_standard_derivatives"] });
    camera = canvasCamera2d(c, {
      target: DEFAULT_TARGET,
      distance: DEFAULT_DISTANCE
    });
    scroll = createScroll(c);
    mousePosition = createMousePos(c);
    mousePressed = createMousePrs(c);

    scroll.on("scroll", () => {
      drawRaf(); // eslint-disable-line
    });
    mousePosition.on("move", () => {
      if (mouseDown) drawRaf(); // eslint-disable-line
    });
    mousePressed.on("down", mouseDownHandler);
    mousePressed.on("up", mouseUpHandler);
  };

  const destroy = () => {
    _canvas = undefined;
    camera = undefined;
    regl = undefined;
    scroll.dispose();
    mousePosition.dispose();
    mousePressed.dispose();
  };

  const canvasGetter = () => _canvas;
  const canvasSetter = newCanvas => {
    _canvas = newCanvas;
    initRegl(_canvas);
  };
  const colorMapGetter = () => _colorMap;
  const colorMapSetter = newColorMap => {
    _colorMap = newColorMap || DEFAULT_COLORMAP;
  };
  const heightGetter = () => _height;
  const heightSetter = newHeight => {
    _height = +newHeight || DEFAULT_HEIGHT;
    _canvas.height = _height * window.devicePixelRatio;
  };
  const paddingGetter = () => _padding;
  const paddingSetter = newPadding => {
    _padding = +newPadding || DEFAULT_PADDING;
    _padding = Math.max(0, Math.min(0.5, _padding));
  };
  const pointSizeGetter = () => _pointSize;
  const pointSizeSetter = newPointSize => {
    _pointSize = +newPointSize || DEFAULT_POINT_SIZE;
  };
  const pointSizeHighlightGetter = () => _pointSizeHighlight;
  const pointSizeHighlightSetter = newPointSizeHighlight => {
    _pointSizeHighlight =
      +newPointSizeHighlight || DEFAULT_POINT_SIZE_HIGHLIGHT;
  };
  const widthGetter = () => _width;
  const widthSetter = newWidth => {
    _width = +newWidth || DEFAULT_WIDTH;
    _canvas.width = _width * window.devicePixelRatio;
  };

  initRegl(_canvas);

  const drawPoints = points =>
    regl({
      frag: FRAG_SHADER,
      vert: VERT_SHADER,

      blend: {
        enable: true,
        func: {
          srcRGB: "src alpha",
          srcAlpha: "one",
          dstRGB: "one minus src alpha",
          dstAlpha: "one minus src alpha"
        }
      },

      depth: { enable: false },

      attributes: {
        // each of these gets mapped to a single entry for each of the points.
        // this means the vertex shader will receive just the relevant value for
        // a given point.
        position: points.map(d => d.slice(0, 2)),
        color: points.map(d => d[2]),
        extraPointSize: points.map(d => d[3] | 0) // eslint-disable-line no-bitwise
      },

      uniforms: {
        // Total area that is being used. Value must be in [0, 1]
        span: regl.prop("span"),
        basePointSize: regl.prop("basePointSize"),
        camera: regl.prop("camera")
      },

      count: points.length,

      primitive: "points"
    });

  const highlightPoints = (points, numHighlights) => {
    const N = points.length;
    const highlightedPoints = [...points];

    for (let i = 0; i < numHighlights; i++) {
      const pt = highlightedPoints[N - numHighlights + i];
      const ptColor = [...pt[2].slice(0, 3)];
      const ptSize = _pointSize * _pointSizeHighlight;
      // Update color and point size to the outer most black outline
      pt[2] = [0, 0, 0, 0.33];
      pt[3] = ptSize + 6;
      // Add second white outline
      highlightedPoints.push([pt[0], pt[1], [1, 1, 1, 1], ptSize + 2]);
      // Finally add the point itself again to be on top
      highlightedPoints.push([pt[0], pt[1], [...ptColor, 1], ptSize]);
    }

    return highlightedPoints;
  };

  const draw = (points = _points, numHighlight = _numHighlight) => {
    _points = points;
    _numHighlight = numHighlight;

    if (points.length === 0) return;

    // clear the buffer
    regl.clear({
      // background color (transparent)
      color: [0, 0, 0, 0],
      depth: 1
    });

    // Update camera
    const isCameraChanged = camera.tick();

    // arguments are available via `regl.prop`.
    drawPoints(highlightPoints(points, numHighlight))({
      span: 1 - _padding,
      basePointSize: _pointSize,
      camera: camera.view()
    });

    // Publish camera change
    if (isCameraChanged) pubSub.publish("camera", camera.position);
  };

  const drawRaf = withRaf(draw);

  const refresh = () => {
    regl.poll();
  };

  const reset = () => {
    camera.lookAt([...DEFAULT_DISTANCE], DEFAULT_DISTANCE);
    drawRaf();
    pubSub.publish("camera", camera.position);
  };

  return {
    get canvas() {
      return canvasGetter();
    },
    set canvas(arg) {
      return canvasSetter(arg);
    },
    get colorMap() {
      return colorMapGetter();
    },
    set colorMap(arg) {
      return colorMapSetter(arg);
    },
    get height() {
      return heightGetter();
    },
    set height(arg) {
      return heightSetter(arg);
    },
    get padding() {
      return paddingGetter();
    },
    set padding(arg) {
      return paddingSetter(arg);
    },
    get pointSize() {
      return pointSizeGetter();
    },
    set pointSize(arg) {
      return pointSizeSetter(arg);
    },
    get pointSizeHighlight() {
      return pointSizeHighlightGetter();
    },
    set pointSizeHighlight(arg) {
      return pointSizeHighlightSetter(arg);
    },
    get width() {
      return widthGetter();
    },
    set width(arg) {
      return widthSetter(arg);
    },
    draw: drawRaf,
    refresh,
    destroy,
    reset,
    subscribe: pubSub.subscribe,
    unsubscribe: pubSub.unsubscribe
  };
};

export default Scatterplot;
