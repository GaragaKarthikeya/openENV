const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./index-CxR0Yf_7.js","./index-BvBk1Iap.js","./index-CrbFh6qt.css"])))=>i.map(i=>d[i]);
import { R as push, aa as onMount, y as user_effect, w as get, a as append, T as pop, a5 as user_derived, u as state, W as from_html, af as __vitePreload, x as set } from './index-BvBk1Iap.js';
import { d as bind_this } from './i18n-CMo5lpzy.js';

var root = from_html(`<canvas></canvas>`);

function Canvas3D($$anchor, $$props) {
	push($$props, true);

	let BABYLON_VIEWER;
	let url = user_derived(() => $$props.value.url);
	let canvas;
	let viewer = state(void 0);
	let viewerDetails = state(void 0);
	let mounted = state(false);

	onMount(() => {
		const initViewer = async () => {
			BABYLON_VIEWER = await __vitePreload(() => import('./index-CxR0Yf_7.js').then(n => n.cx),true              ?__vite__mapDeps([0,1,2]):void 0,import.meta.url);

			BABYLON_VIEWER.CreateViewerForCanvas(canvas, {
				clearColor: $$props.clear_color,
				useRightHandedSystem: true,
				animationAutoPlay: true,
				cameraAutoOrbit: { enabled: false },
				onInitialized: (details) => {
					set(viewerDetails, details, true);
				}
			}).then((promiseViewer) => {
				set(viewer, promiseViewer, true);
				set(mounted, true);
			});
		};

		initViewer();

		return () => {
			get(viewer)?.dispose();
		};
	});

	user_effect(() => {
		if (get(mounted)) {
			load_model(get(url));
		}
	});

	function setRenderingMode(pointsCloud, wireframe) {
		if (!get(viewerDetails)) return;

		get(viewerDetails).scene.forcePointsCloud = pointsCloud;
		get(viewerDetails).scene.forceWireframe = wireframe;
	}

	function load_model(url) {
		if (get(viewer)) {
			if (url) {
				get(viewer).loadModel(url, { pluginOptions: { obj: { importVertexColors: true } } }).then(() => {
					if ($$props.display_mode === "point_cloud") {
						setRenderingMode(true, false);
					} else if ($$props.display_mode === "wireframe") {
						setRenderingMode(false, true);
					} else {
						update_camera($$props.camera_position, $$props.zoom_speed, $$props.pan_speed);
					}
				});
			} else {
				get(viewer).resetModel();
			}
		}
	}

	function update_camera(camera_position, zoom_speed, pan_speed) {
		if (!get(viewerDetails)) return;

		const camera = get(viewerDetails).camera;

		if (camera_position[0] !== null) {
			camera.alpha = camera_position[0] * Math.PI / 180;
		}

		if (camera_position[1] !== null) {
			camera.beta = camera_position[1] * Math.PI / 180;
		}

		if (camera_position[2] !== null) {
			camera.radius = camera_position[2];
		}

		camera.lowerRadiusLimit = 0.1;

		const updateCameraSensibility = () => {
			camera.wheelPrecision = 250 / (camera.radius * zoom_speed);
			camera.panningSensibility = 10000 * pan_speed / camera.radius;
		};

		updateCameraSensibility();
		camera.onAfterCheckInputsObservable.add(updateCameraSensibility);
	}

	function reset_camera_position() {
		if (get(viewerDetails) && get(viewer)) {
			get(viewer).resetCamera();
		}
	}

	var $$exports = { update_camera, reset_camera_position };
	var canvas_1 = root();

	bind_this(canvas_1, ($$value) => canvas = $$value, () => canvas);
	append($$anchor, canvas_1);

	return pop($$exports);
}

export { Canvas3D as default };
//# sourceMappingURL=Canvas3D-e6PmZQ-C.js.map
