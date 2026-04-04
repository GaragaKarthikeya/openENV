const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./Canvas3D-e6PmZQ-C.js","./index-BvBk1Iap.js","./index-CrbFh6qt.css","./i18n-CMo5lpzy.js","./i18n-CGFL4PZH.css","./Canvas3DGS-UwN0Q52I.js"])))=>i.map(i=>d[i]);
import { R as push, u as state, v as proxy, y as user_effect, x as set, w as get, S as first_child, a as append, T as pop, af as __vitePreload, a5 as user_derived, X as sibling, W as from_html, V as child, Y as reset, t as template_effect, a6 as comment } from './index-BvBk1Iap.js';
import { p as prop, i as if_block, a as set_attribute, d as bind_this } from './i18n-CMo5lpzy.js';
import { I as IconButton, c as component } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { B as BlockLabel } from './BlockLabel-oGSqUii_.js';
import { D as Download } from './Download-pFhUaTl4.js';
import { F as File } from './File-_cC4Z_uD.js';
import { U as Undo } from './Undo-BxcJDM8E.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { d as dequal } from './index-DnoGeqVF.js';
import './snippet-Bl5YVoCg.js';

var root_2 = from_html(`<!> <a><!></a>`, 1);
var root_1 = from_html(`<div class="model3D svelte-pnaihf" data-testid="model3d"><!> <!></div>`);
var root = from_html(`<!> <!>`, 1);

function Model3D($$anchor, $$props) {
	push($$props, true);

	let display_mode = prop($$props, 'display_mode', 3, "solid"),
		clear_color = prop($$props, 'clear_color', 19, () => [0, 0, 0, 0]),
		label = prop($$props, 'label', 3, ""),
		zoom_speed = prop($$props, 'zoom_speed', 3, 1),
		pan_speed = prop($$props, 'pan_speed', 3, 1),
		camera_position = prop($$props, 'camera_position', 19, () => [null, null, null]),
		has_change_history = prop($$props, 'has_change_history', 3, false);

	let current_settings = state(proxy({
		camera_position: camera_position(),
		zoom_speed: zoom_speed(),
		pan_speed: pan_speed()
	}));

	let use_3dgs = state(false);
	let Canvas3DGSComponent = state(void 0);
	let Canvas3DComponent = state(void 0);
	let canvas3d = state(void 0);

	async function loadCanvas3D() {
		const module = await __vitePreload(() => import('./Canvas3D-e6PmZQ-C.js'),true              ?__vite__mapDeps([0,1,2,3,4]):void 0,import.meta.url);

		return module.default;
	}

	async function loadCanvas3DGS() {
		const module = await __vitePreload(() => import('./Canvas3DGS-UwN0Q52I.js'),true              ?__vite__mapDeps([5,3,1,2,4]):void 0,import.meta.url);

		return module.default;
	}

	user_effect(() => {
		if ($$props.value) {
			set(use_3dgs, $$props.value.path.endsWith(".splat") || $$props.value.path.endsWith(".ply"), true);

			if (get(use_3dgs)) {
				loadCanvas3DGS().then((component) => {
					set(Canvas3DGSComponent, component, true);
				});
			} else {
				loadCanvas3D().then((component) => {
					set(Canvas3DComponent, component, true);
				});
			}
		}
	});

	function handle_undo() {
		get(canvas3d)?.reset_camera_position();
	}

	user_effect(() => {
		if (!dequal(get(current_settings).camera_position, camera_position()) || get(current_settings).zoom_speed !== zoom_speed() || get(current_settings).pan_speed !== pan_speed()) {
			get(canvas3d)?.update_camera(camera_position(), zoom_speed(), pan_speed());

			set(
				current_settings,
				{
					camera_position: camera_position(),
					zoom_speed: zoom_speed(),
					pan_speed: pan_speed()
				},
				true
			);
		}
	});

	var fragment = root();
	var node = first_child(fragment);

	{
		let $0 = user_derived(() => label() || $$props.i18n("3D_model.3d_model"));

		BlockLabel(node, {
			get show_label() {
				return $$props.show_label;
			},

			get Icon() {
				return File;
			},

			get label() {
				return get($0);
			}
		});
	}

	var node_1 = sibling(node, 2);

	{
		var consequent_2 = ($$anchor) => {
			var div = root_1();
			var node_2 = child(div);

			IconButtonWrapper(node_2, {
				children: ($$anchor, $$slotProps) => {
					var fragment_1 = root_2();
					var node_3 = first_child(fragment_1);

					{
						var consequent = ($$anchor) => {
							{
								let $0 = user_derived(() => !has_change_history());

								IconButton($$anchor, {
									get Icon() {
										return Undo;
									},
									label: 'Undo',
									onclick: () => handle_undo(),
									get disabled() {
										return get($0);
									}
								});
							}
						};

						if_block(node_3, ($$render) => {
							if (!get(use_3dgs)) $$render(consequent);
						});
					}

					var a = sibling(node_3, 2);

					set_attribute(a, 'target', window.__is_colab__ ? "_blank" : null);

					var node_4 = child(a);

					{
						let $0 = user_derived(() => $$props.i18n("common.download"));

						IconButton(node_4, {
							get Icon() {
								return Download;
							},

							get label() {
								return get($0);
							}
						});
					}

					reset(a);

					template_effect(() => {
						set_attribute(a, 'href', $$props.value.url);
						set_attribute(a, 'download', window.__is_colab__ ? null : $$props.value.orig_name || $$props.value.path);
					});

					append($$anchor, fragment_1);
				},
				$$slots: { default: true }
			});

			var node_5 = sibling(node_2, 2);

			{
				var consequent_1 = ($$anchor) => {
					var fragment_3 = comment();
					var node_6 = first_child(fragment_3);

					component(node_6, () => get(Canvas3DGSComponent), ($$anchor, $$component) => {
						$$component($$anchor, {
							get value() {
								return $$props.value;
							},

							get zoom_speed() {
								return zoom_speed();
							},

							get pan_speed() {
								return pan_speed();
							}
						});
					});

					append($$anchor, fragment_3);
				};

				var alternate = ($$anchor) => {
					var fragment_4 = comment();
					var node_7 = first_child(fragment_4);

					component(node_7, () => get(Canvas3DComponent), ($$anchor, $$component) => {
						bind_this(
							$$component($$anchor, {
								get value() {
									return $$props.value;
								},

								get display_mode() {
									return display_mode();
								},

								get clear_color() {
									return clear_color();
								},

								get camera_position() {
									return camera_position();
								},

								get zoom_speed() {
									return zoom_speed();
								},

								get pan_speed() {
									return pan_speed();
								}
							}),
							($$value) => set(canvas3d, $$value, true),
							() => get(canvas3d)
						);
					});

					append($$anchor, fragment_4);
				};

				if_block(node_5, ($$render) => {
					if (get(use_3dgs)) $$render(consequent_1); else $$render(alternate, false);
				});
			}

			reset(div);
			append($$anchor, div);
		};

		if_block(node_1, ($$render) => {
			if ($$props.value) $$render(consequent_2);
		});
	}

	append($$anchor, fragment);
	pop();
}

export { Model3D as default };
//# sourceMappingURL=Model3D-CWtA1JFE.js.map
