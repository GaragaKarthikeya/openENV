const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./Canvas3D-e6PmZQ-C.js","./index-BvBk1Iap.js","./index-CrbFh6qt.css","./i18n-CMo5lpzy.js","./i18n-CGFL4PZH.css","./Canvas3DGS-UwN0Q52I.js"])))=>i.map(i=>d[i]);
import { p as prop, i as if_block, s as slot, d as bind_this, r as rest_props, g as spread_props } from './i18n-CMo5lpzy.js';
import { R as push, y as user_effect, x as set, w as get, S as first_child, a as append, T as pop, u as state, af as __vitePreload, a5 as user_derived, X as sibling, W as from_html, a6 as comment, V as child, Y as reset, M as tick, v as proxy } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import Model3D from './Model3D-CWtA1JFE.js';
import { c as component } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { U as Upload } from './Upload-CPeZiqam.js';
import { M as ModifyUpload } from './ModifyUpload-B5K-5Hqx.js';
import { B as BlockLabel } from './BlockLabel-oGSqUii_.js';
import { F as File } from './File-_cC4Z_uD.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { E as Empty } from './Empty-CHa-gxv7.js';
import { U as UploadText } from './UploadText-BSD1IoCI.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
export { default as BaseExample } from './Example-qy-BJ3UM.js';
import './clone-DaB-S2nH.js';
import './Download-pFhUaTl4.js';
import './Undo-BxcJDM8E.js';
import './index-DnoGeqVF.js';
import './snippet-Bl5YVoCg.js';
import './actions-Cit_UA5S.js';
import './DownloadLink-KvrFKyMn.js';
import './Clear-Dgs5UhTO.js';
import './Edit-Dr58YXBX.js';
import './html-CdUEtR5E.js';
import './index-C7e-J7CF.js';

var root_4$1 = from_html(`<div class="input-model svelte-18wa0f8"><!> <!></div>`);
var root_1 = from_html(`<!> <!>`, 1);

function Model3DUpload($$anchor, $$props) {
	push($$props, true);

	let value = prop($$props, 'value', 15),
		display_mode = prop($$props, 'display_mode', 3, "solid"),
		clear_color = prop($$props, 'clear_color', 19, () => [0, 0, 0, 0]),
		label = prop($$props, 'label', 3, ""),
		zoom_speed = prop($$props, 'zoom_speed', 3, 1),
		pan_speed = prop($$props, 'pan_speed', 3, 1),
		max_file_size = prop($$props, 'max_file_size', 3, null),
		uploading = prop($$props, 'uploading', 15),
		upload_promise = prop($$props, 'upload_promise', 15),
		camera_position = prop($$props, 'camera_position', 19, () => [null, null, null]);

	let use_3dgs = state(false);
	let Canvas3DGSComponent = state(void 0);
	let Canvas3DComponent = state(void 0);
	let canvas3d = state(void 0);
	let dragging = state(false);

	async function loadCanvas3D() {
		const module = await __vitePreload(() => import('./Canvas3D-e6PmZQ-C.js'),true              ?__vite__mapDeps([0,1,2,3,4]):void 0,import.meta.url);

		return module.default;
	}

	async function loadCanvas3DGS() {
		const module = await __vitePreload(() => import('./Canvas3DGS-UwN0Q52I.js'),true              ?__vite__mapDeps([5,3,1,2,4]):void 0,import.meta.url);

		return module.default;
	}

	user_effect(() => {
		if (value()) {
			set(use_3dgs, value().path.endsWith(".splat") || value().path.endsWith(".ply"), true);

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

	user_effect(() => {
		$$props.ondrag?.(get(dragging));
	});

	async function handle_upload(detail) {
		value(detail);
		await tick();
		$$props.onchange?.(value());
		$$props.onload?.(value());
	}

	async function handle_clear() {
		value(null);
		await tick();
		$$props.onclear?.();
		$$props.onchange?.(null);
	}

	async function handle_undo() {
		get(canvas3d)?.reset_camera_position();
	}

	function handle_error(error) {
		$$props.onerror?.(error);
	}

	var fragment = root_1();
	var node = first_child(fragment);

	{
		let $0 = user_derived(() => label() || "3D Model");

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
		var consequent = ($$anchor) => {
			{
				let $0 = user_derived(() => $$props.i18n("model3d.drop_to_upload"));

				Upload($$anchor, {
					get upload() {
						return $$props.upload;
					},

					get stream_handler() {
						return $$props.stream_handler;
					},
					onload: handle_upload,
					get root() {
						return $$props.root;
					},

					get max_file_size() {
						return max_file_size();
					},

					filetype: [
						".stl",
						".obj",
						".gltf",
						".glb",
						"model/obj",
						".splat",
						".ply"
					],
					onerror: handle_error,
					get aria_label() {
						return get($0);
					},

					get upload_promise() {
						return upload_promise();
					},

					set upload_promise($$value) {
						upload_promise($$value);
					},

					get dragging() {
						return get(dragging);
					},

					set dragging($$value) {
						set(dragging, $$value, true);
					},

					get uploading() {
						return uploading();
					},

					set uploading($$value) {
						uploading($$value);
					},

					children: ($$anchor, $$slotProps) => {
						var fragment_2 = comment();
						var node_2 = first_child(fragment_2);

						slot(node_2, $$props, 'default', {});
						append($$anchor, fragment_2);
					},
					$$slots: { default: true }
				});
			}
		};

		var alternate_1 = ($$anchor) => {
			var div = root_4$1();
			var node_3 = child(div);

			{
				let $0 = user_derived(() => !get(use_3dgs));

				ModifyUpload(node_3, {
					get undoable() {
						return get($0);
					},
					onclear: handle_clear,
					get i18n() {
						return $$props.i18n;
					},
					onundo: handle_undo
				});
			}

			var node_4 = sibling(node_3, 2);

			{
				var consequent_1 = ($$anchor) => {
					var fragment_3 = comment();
					var node_5 = first_child(fragment_3);

					component(node_5, () => get(Canvas3DGSComponent), ($$anchor, $$component) => {
						$$component($$anchor, {
							get value() {
								return value();
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
					var node_6 = first_child(fragment_4);

					component(node_6, () => get(Canvas3DComponent), ($$anchor, $$component) => {
						bind_this(
							$$component($$anchor, {
								get value() {
									return value();
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

				if_block(node_4, ($$render) => {
					if (get(use_3dgs)) $$render(consequent_1); else $$render(alternate, false);
				});
			}

			reset(div);
			append($$anchor, div);
		};

		if_block(node_1, ($$render) => {
			if (value() == null) $$render(consequent); else $$render(alternate_1, false);
		});
	}

	append($$anchor, fragment);
	pop();
}

var root_4 = from_html(`<!> <!> <!>`, 1);
var root_2 = from_html(`<!> <!>`, 1);
var root_8 = from_html(`<!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	class Model3dGradio extends Gradio {
		async get_data() {
			if (get(upload_promise)) {
				await get(upload_promise);
				await tick();
			}

			const data = await super.get_data();

			return data;
		}
	}

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Model3dGradio(props);
	let old_value = state(proxy(gradio.props.value));
	let uploading = state(false);
	let dragging = state(false);
	let has_change_history = state(false);
	let upload_promise = state(void 0);
	const is_browser = typeof window !== "undefined";

	user_effect(() => {
		if (get(old_value) !== gradio.props.value) {
			set(old_value, gradio.props.value, true);
			gradio.dispatch("change");
		}
	});

	function handle_change(detail) {
		gradio.props.value = detail;
		gradio.dispatch("change", detail);
		set(has_change_history, true);
	}

	function handle_drag(detail) {
		set(dragging, detail, true);
	}

	function handle_clear() {
		gradio.props.value = null;
		gradio.dispatch("clear");
	}

	function handle_load(detail) {
		gradio.props.value = detail;
		gradio.dispatch("upload");
	}

	function handle_error(detail) {
		if (gradio.shared.loading_status) gradio.shared.loading_status.status = "error";

		gradio.dispatch("error", detail);
	}

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent_2 = ($$anchor) => {
			{
				let $0 = user_derived(() => gradio.props.value === null ? "dashed" : "solid");
				let $1 = user_derived(() => get(dragging) ? "focus" : "base");

				Block($$anchor, {
					get visible() {
						return gradio.shared.visible;
					},

					get variant() {
						return get($0);
					},

					get border_mode() {
						return get($1);
					},
					padding: false,
					get elem_id() {
						return gradio.shared.elem_id;
					},

					get elem_classes() {
						return gradio.shared.elem_classes;
					},

					get container() {
						return gradio.shared.container;
					},

					get scale() {
						return gradio.shared.scale;
					},

					get min_width() {
						return gradio.shared.min_width;
					},

					get height() {
						return gradio.props.height;
					},

					children: ($$anchor, $$slotProps) => {
						var fragment_2 = root_2();
						var node_1 = first_child(fragment_2);

						Static(node_1, spread_props(
							{
								get autoscroll() {
									return gradio.shared.autoscroll;
								},

								get i18n() {
									return gradio.i18n;
								}
							},
							() => gradio.shared.loading_status,
							{
								on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
							}
						));

						var node_2 = sibling(node_1, 2);

						{
							var consequent = ($$anchor) => {
								Model3D($$anchor, {
									get value() {
										return gradio.props.value;
									},

									get i18n() {
										return gradio.i18n;
									},

									get display_mode() {
										return gradio.props.display_mode;
									},

									get clear_color() {
										return gradio.props.clear_color;
									},

									get label() {
										return gradio.shared.label;
									},

									get show_label() {
										return gradio.shared.show_label;
									},

									get camera_position() {
										return gradio.props.camera_position;
									},

									get zoom_speed() {
										return gradio.props.zoom_speed;
									},

									get has_change_history() {
										return get(has_change_history);
									}
								});
							};

							var alternate = ($$anchor) => {
								var fragment_4 = root_4();
								var node_3 = first_child(fragment_4);

								{
									var consequent_1 = ($$anchor) => {
										IconButtonWrapper($$anchor, {
											get buttons() {
												return gradio.props.buttons;
											},

											on_custom_button_click: (id) => {
												gradio.dispatch("custom_button_click", { id });
											}
										});
									};

									if_block(node_3, ($$render) => {
										if (gradio.shared.show_label && gradio.props.buttons && gradio.props.buttons.length > 0) $$render(consequent_1);
									});
								}

								var node_4 = sibling(node_3, 2);

								{
									let $0 = user_derived(() => gradio.shared.label || "3D Model");

									BlockLabel(node_4, {
										get show_label() {
											return gradio.shared.show_label;
										},

										get Icon() {
											return File;
										},

										get label() {
											return get($0);
										}
									});
								}

								var node_5 = sibling(node_4, 2);

								Empty(node_5, {
									unpadded_box: true,
									size: 'large',
									children: ($$anchor, $$slotProps) => {
										File($$anchor);
									},
									$$slots: { default: true }
								});

								append($$anchor, fragment_4);
							};

							if_block(node_2, ($$render) => {
								if (gradio.props.value && is_browser) $$render(consequent); else $$render(alternate, false);
							});
						}

						append($$anchor, fragment_2);
					},
					$$slots: { default: true }
				});
			}
		};

		var alternate_1 = ($$anchor) => {
			{
				let $0 = user_derived(() => gradio.props.value === null ? "dashed" : "solid");
				let $1 = user_derived(() => get(dragging) ? "focus" : "base");

				Block($$anchor, {
					get visible() {
						return gradio.shared.visible;
					},

					get variant() {
						return get($0);
					},

					get border_mode() {
						return get($1);
					},
					padding: false,
					get elem_id() {
						return gradio.shared.elem_id;
					},

					get elem_classes() {
						return gradio.shared.elem_classes;
					},

					get container() {
						return gradio.shared.container;
					},

					get scale() {
						return gradio.shared.scale;
					},

					get min_width() {
						return gradio.shared.min_width;
					},

					get height() {
						return gradio.props.height;
					},

					children: ($$anchor, $$slotProps) => {
						var fragment_8 = root_8();
						var node_6 = first_child(fragment_8);

						Static(node_6, spread_props(
							{
								get autoscroll() {
									return gradio.shared.autoscroll;
								},

								get i18n() {
									return gradio.i18n;
								}
							},
							() => gradio.shared.loading_status,
							{
								on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
							}
						));

						var node_7 = sibling(node_6, 2);

						Model3DUpload(node_7, {
							get label() {
								return gradio.shared.label;
							},

							get show_label() {
								return gradio.shared.show_label;
							},

							get root() {
								return gradio.shared.root;
							},

							get display_mode() {
								return gradio.props.display_mode;
							},

							get clear_color() {
								return gradio.props.clear_color;
							},

							get camera_position() {
								return gradio.props.camera_position;
							},

							get zoom_speed() {
								return gradio.props.zoom_speed;
							},
							onchange: handle_change,
							ondrag: handle_drag,
							onclear: handle_clear,
							onload: handle_load,
							onerror: handle_error,
							get i18n() {
								return gradio.i18n;
							},

							get max_file_size() {
								return gradio.shared.max_file_size;
							},
							upload: (...args) => gradio.shared.client.upload(...args),
							stream_handler: (...args) => gradio.shared.client.stream(...args),
							get upload_promise() {
								return get(upload_promise);
							},

							set upload_promise($$value) {
								set(upload_promise, $$value, true);
							},

							get value() {
								return gradio.props.value;
							},

							set value($$value) {
								gradio.props.value = $$value;
							},

							get uploading() {
								return get(uploading);
							},

							set uploading($$value) {
								set(uploading, $$value, true);
							},

							children: ($$anchor, $$slotProps) => {
								UploadText($$anchor, {
									get i18n() {
										return gradio.i18n;
									},
									type: 'file'
								});
							},
							$$slots: { default: true }
						});

						append($$anchor, fragment_8);
					},
					$$slots: { default: true }
				});
			}
		};

		if_block(node, ($$render) => {
			if (!gradio.shared.interactive) $$render(consequent_2); else $$render(alternate_1, false);
		});
	}

	append($$anchor, fragment);
	pop();
}

export { Model3D as BaseModel3D, Model3DUpload as BaseModel3DUpload, Index as default };
//# sourceMappingURL=Index-VR1jY_B8.js.map
