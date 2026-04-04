import { g as spread_props, i as if_block, r as rest_props, d as bind_this, h as clsx, b as set_class } from './i18n-CMo5lpzy.js';
import { R as push, u as state, v as proxy, S as first_child, a as append, x as set, w as get, T as pop, M as tick, a5 as user_derived, X as sibling, W as from_html, t as template_effect, V as child, Y as reset, a6 as comment } from './index-BvBk1Iap.js';
import { b as bubble_event } from './misc-ByVveAau.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { S as SelectSource } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { U as UploadText } from './UploadText-BSD1IoCI.js';
import { G as Gallery, h as handle_save } from './Gallery-DaBd1NXs.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import { a as FileUpload } from './FileUpload-9lL-iwhh.js';
/* empty css                                               */
/* empty css                                             */
/* empty css                                                    */
import { W as Webcam } from './Webcam-rWLs1V82.js';
import './Upload-CPeZiqam.js';
/* empty css                                                     */
/* empty css                                               */
export { default as BaseExample } from './Example-BgRaJ5Hk.js';
import './snippet-Bl5YVoCg.js';
import './clone-DaB-S2nH.js';
import './window-BGXjFJtG.js';
import './event-modifiers-DanhKw3_.js';
import './BlockLabel-oGSqUii_.js';
import './Empty-CHa-gxv7.js';
import './ShareButton-Dz0y1C1K.js';
import './Clear-Dgs5UhTO.js';
import './Download-pFhUaTl4.js';
import './Image-C3F5KG1F.js';
import './Play-CWcFSzc7.js';
import './IconButtonWrapper-Ba0GidgU.js';
import './FullscreenButton-CNmhk6nd.js';
import './Maximize-B_43ylTs.js';
import './ModifyUpload-B5K-5Hqx.js';
import './DownloadLink-KvrFKyMn.js';
import './Edit-Dr58YXBX.js';
import './Undo-BxcJDM8E.js';
import './Image-CDw5qb5O.js';
import './Video-B3xDM6GN.js';
import './actions-Cit_UA5S.js';
import './index-DnoGeqVF.js';
import './html-CdUEtR5E.js';
import './index-C7e-J7CF.js';
import './File-_cC4Z_uD.js';
import './DropdownArrow-Dw1NZKCM.js';
import './Square-B6INQV79.js';
import './Spinner-BrK-6emq.js';
import './StreamingBar-bVF1kkqe.js';

var root_2 = from_html(`<div><!></div> <!> <!>`, 1);
var root_1 = from_html(`<!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	let upload_promise = state(void 0);

	class GalleryGradio extends Gradio {
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
	const gradio = new GalleryGradio(props, { selected_index: null, file_types: ["image", "video"] });
	let fullscreen = state(false);

	function handle_delete(event) {
		if (!gradio.props.value) return;

		const { index } = event.detail;

		gradio.dispatch("delete", event.detail);
		gradio.props.value = gradio.props.value.filter((_, i) => i !== index);
		gradio.dispatch("change", gradio.props.value);
	}

	async function process_upload_files(files) {
		const processed_files = await Promise.all(files.map(async (x) => {
			if (x.path?.toLowerCase().endsWith(".svg") && x.url) {
				const response = await fetch(x.url);
				const svgContent = await response.text();

				return {
					...x,
					url: `data:image/svg+xml,${encodeURIComponent(svgContent)}`
				};
			}

			return x;
		}));

		return processed_files.map((x) => x.mime_type?.includes("video")
			? { video: x, caption: null }
			: { image: x, caption: null });
	}

	let upload_input;
	let active_source = state(proxy(gradio.props.sources ? gradio.props.sources[0] : "upload"));
	let no_value = user_derived(() => gradio.props.value === null ? true : gradio.props.value.length === 0);

	let sources = user_derived(() => {
		if (gradio.props.file_types?.includes("video") && gradio.props.sources.includes("webcam")) {
			return gradio.props.sources.concat(["webcam-video"]);
		} else {
			return gradio.props.sources;
		}
	});

	async function paste_clipboard() {
		navigator.clipboard.read().then(async (items) => {
			let file = null;

			for (let i = 0; i < items.length; i++) {
				const type = items[i].types.find((t) => (gradio.props.file_types || ["image"]).some((ft) => t.startsWith(ft + "/")));

				if (type) {
					const blob = await items[i].getType(type);

					file = new File([blob], `clipboard.${type.replace("image/", "")}`);

					break;
				}
			}

			if (file) {
				const f = await handle_save(file, (f) => gradio.shared.client.upload(f, gradio.shared.root), "clipboard_upload");
				const processed_files = await process_upload_files(f);

				gradio.props.value?.push(...processed_files);
				gradio.dispatch("change", gradio.props.value);
				set(active_source, null);
			} else {
				gradio.dispatch("warning", "No image or video found in clipboard");
			}
		});
	}

	async function handle_select_source(source) {
		switch (source) {
			case "clipboard":
				await paste_clipboard();
				break;
		}
	}

	async function onsource_change(source) {
		await tick();

		if (source === "clipboard") {
			await paste_clipboard();
		} else {
			set(active_source, source, true);
			set(no_value, true);
		}
	}

	{
		let $0 = user_derived(() => gradio.props.height || undefined);

		Block($$anchor, {
			get visible() {
				return gradio.shared.visible;
			},
			variant: 'solid',
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
			allow_overflow: false,
			get height() {
				return get($0);
			},

			get fullscreen() {
				return get(fullscreen);
			},

			set fullscreen($$value) {
				set(fullscreen, $$value, true);
			},

			children: ($$anchor, $$slotProps) => {
				var fragment_1 = root_1();
				var node = first_child(fragment_1);

				Static(node, spread_props(
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

				var node_1 = sibling(node, 2);

				{
					var consequent_3 = ($$anchor) => {
						var fragment_2 = root_2();
						var div = first_child(fragment_2);
						var node_2 = child(div);

						bind_this(
							FileUpload(node_2, {
								value: null,
								get root() {
									return gradio.shared.root;
								},

								get label() {
									return gradio.shared.label;
								},

								get max_file_size() {
									return gradio.shared.max_file_size;
								},
								file_count: "multiple",
								get file_types() {
									return gradio.props.file_types;
								},

								get i18n() {
									return gradio.i18n;
								},
								upload: (...args) => gradio.shared.client.upload(...args),
								stream_handler: (...args) => gradio.shared.client.stream(...args),
								onupload: async (e) => {
									const files = Array.isArray(e) ? e : [e];

									gradio.props.value = await process_upload_files(files);
									set(active_source, null);
									gradio.dispatch("upload", gradio.props.value);
									gradio.dispatch("change", gradio.props.value);
								},

								onerror: ({ detail }) => {
									gradio.shared.loading_status = gradio.shared.loading_status || {};
									gradio.shared.loading_status.status = "error";
									gradio.dispatch("error", detail);
								},

								get upload_promise() {
									return get(upload_promise);
								},

								set upload_promise($$value) {
									set(upload_promise, $$value, true);
								},

								children: ($$anchor, $$slotProps) => {
									UploadText($$anchor, {
										get i18n() {
											return gradio.i18n;
										},
										type: 'gallery'
									});
								},
								$$slots: { default: true }
							}),
							($$value) => upload_input = $$value,
							() => upload_input
						);

						reset(div);

						var node_3 = sibling(div, 2);

						{
							var consequent = ($$anchor) => {
								Webcam($$anchor, {
									get root() {
										return gradio.shared.root;
									},
									value: null,
									mirror_webcam: true,
									streaming: false,
									mode: 'image',
									include_audio: false,
									get i18n() {
										return gradio.i18n;
									},
									upload: (...args) => gradio.shared.client.upload(...args),
									$$events: {
										capture: async (e) => {
											const f = await handle_save(e.detail, (f) => gradio.shared.client.upload(f, gradio.shared.root), "webcam_upload");
											const processed_files = await process_upload_files(f);

											gradio.props.value?.push(...processed_files);
											set(active_source, null);
											gradio.dispatch("change", gradio.props.value);
										},

										error($$arg) {
											bubble_event.call(this, $$props, $$arg);
										},

										drag($$arg) {
											bubble_event.call(this, $$props, $$arg);
										},

										close_stream($$arg) {
											bubble_event.call(this, $$props, $$arg);
										}
									}
								});
							};

							var alternate = ($$anchor) => {
								var fragment_5 = comment();
								var node_4 = first_child(fragment_5);

								{
									var consequent_1 = ($$anchor) => {
										Webcam($$anchor, {
											get root() {
												return gradio.shared.root;
											},
											value: null,
											mirror_webcam: true,
											streaming: false,
											mode: 'video',
											include_audio: false,
											get i18n() {
												return gradio.i18n;
											},
											upload: (...args) => gradio.shared.client.upload(...args),
											$$events: {
												capture: async (e) => {
													const f = { ...e.detail };

													f.mime_type = "video/webm";

													const processed_files = await process_upload_files([f]);

													gradio.props.value?.push(...processed_files);
													set(active_source, null);
													gradio.dispatch("change", gradio.props.value);
												},

												error($$arg) {
													bubble_event.call(this, $$props, $$arg);
												},

												drag($$arg) {
													bubble_event.call(this, $$props, $$arg);
												},

												close_stream($$arg) {
													bubble_event.call(this, $$props, $$arg);
												}
											}
										});
									};

									if_block(
										node_4,
										($$render) => {
											if (get(active_source) === "webcam-video") $$render(consequent_1);
										},
										true
									);
								}

								append($$anchor, fragment_5);
							};

							if_block(node_3, ($$render) => {
								if (get(active_source) === "webcam") $$render(consequent); else $$render(alternate, false);
							});
						}

						var node_5 = sibling(node_3, 2);

						{
							var consequent_2 = ($$anchor) => {
								SelectSource($$anchor, {
									get sources() {
										return get(sources);
									},
									handle_clear: () => gradio.dispatch("clear"),
									handle_select: handle_select_source,
									get active_source() {
										return get(active_source);
									},

									set active_source($$value) {
										set(active_source, $$value, true);
									}
								});
							};

							if_block(node_5, ($$render) => {
								if (get(sources).length > 1 || get(sources).includes("clipboard")) $$render(consequent_2);
							});
						}

						template_effect(($0) => set_class(div, 1, $0, 'svelte-vrqwbn'), [
							() => clsx(!gradio.props.value || get(active_source) && get(active_source).includes("webcam") ? "hidden-upload-input" : "upload-wrapper")
						]);

						append($$anchor, fragment_2);
					};

					var alternate_1 = ($$anchor) => {
						{
							let $0 = user_derived(() => gradio.props.buttons.some((btn) => typeof btn === "string" && btn === "share"));
							let $1 = user_derived(() => gradio.props.buttons.some((btn) => typeof btn === "string" && btn === "download"));
							let $2 = user_derived(() => gradio.props.buttons.some((btn) => typeof btn === "string" && btn === "fullscreen"));

							Gallery($$anchor, {
								onchange: () => gradio.dispatch("change"),
								onclear: () => gradio.dispatch("change"),
								onselect: (e) => gradio.dispatch("select", e),
								onshare: (e) => gradio.dispatch("share", e.detail),
								onerror: (e) => gradio.dispatch("error", e.detail),
								onpreview_open: () => {
									gradio.dispatch("preview_open");
								},
								onpreview_close: () => gradio.dispatch("preview_close"),
								onfullscreen: ({ detail }) => {
									set(fullscreen, detail, true);
								},
								ondelete: handle_delete,
								onupload: async (e) => {
									const files = Array.isArray(e) ? e : [e];
									const new_value = await process_upload_files(files);

									gradio.props.value = gradio.props.value ? [...gradio.props.value, ...new_value] : new_value;
									gradio.dispatch("upload", new_value);
									gradio.dispatch("change", gradio.props.value);
								},

								get sources() {
									return get(sources);
								},
								onsource_change,
								get label() {
									return gradio.shared.label;
								},

								get show_label() {
									return gradio.shared.show_label;
								},

								get columns() {
									return gradio.props.columns;
								},

								get rows() {
									return gradio.props.rows;
								},

								get height() {
									return gradio.props.height;
								},

								get preview() {
									return gradio.props.preview;
								},

								get object_fit() {
									return gradio.props.object_fit;
								},

								get interactive() {
									return gradio.shared.interactive;
								},

								get allow_preview() {
									return gradio.props.allow_preview;
								},

								get show_share_button() {
									return get($0);
								},

								get show_download_button() {
									return get($1);
								},

								get fit_columns() {
									return gradio.props.fit_columns;
								},

								get i18n() {
									return gradio.i18n;
								},
								_fetch: (...args) => gradio.shared.client.fetch(...args),
								get show_fullscreen_button() {
									return get($2);
								},

								get buttons() {
									return gradio.props.buttons;
								},

								on_custom_button_click: (id) => {
									gradio.dispatch("custom_button_click", { id });
								},

								get fullscreen() {
									return get(fullscreen);
								},

								get root() {
									return gradio.shared.root;
								},

								get file_types() {
									return gradio.props.file_types;
								},

								get max_file_size() {
									return gradio.shared.max_file_size;
								},
								upload: (...args) => gradio.shared.client.upload(...args),
								stream_handler: (...args) => gradio.shared.client.stream(...args),
								get selected_index() {
									return gradio.props.selected_index;
								},

								set selected_index($$value) {
									gradio.props.selected_index = $$value;
								},

								get value() {
									return gradio.props.value;
								},

								set value($$value) {
									gradio.props.value = $$value;
								}
							});
						}
					};

					if_block(node_1, ($$render) => {
						if (gradio.shared.interactive && get(no_value)) $$render(consequent_3); else $$render(alternate_1, false);
					});
				}

				append($$anchor, fragment_1);
			},
			$$slots: { default: true }
		});
	}

	pop();
}

export { Gallery as BaseGallery, Index as default };
//# sourceMappingURL=Index-CGOALv4G.js.map
