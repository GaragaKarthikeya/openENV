import { p as prop, i as if_block, r as rest_props, g as spread_props } from './i18n-CMo5lpzy.js';
import { R as push, y as user_effect, w as get, S as first_child, x as set, a as append, T as pop, a5 as user_derived, u as state, X as sibling, W as from_html, V as child, Y as reset, a6 as comment, t as template_effect, a0 as set_text, M as tick } from './index-BvBk1Iap.js';
import { k as key } from './key-BV46G5Fv.js';
import { s as snippet } from './snippet-Bl5YVoCg.js';
import { U as Upload } from './Upload-CPeZiqam.js';
import { V as Video, S as SelectSource } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { B as BlockLabel } from './BlockLabel-oGSqUii_.js';
/* empty css                                             */
/* empty css                                                    */
import { W as Webcam } from './Webcam-rWLs1V82.js';
/* empty css                                                     */
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
/* empty css                                               */
import { p as prettyBytes } from './Video-B3xDM6GN.js';
export { l as loaded, a as playable } from './Video-B3xDM6GN.js';
import { P as Player, V as VideoPreview } from './VideoPreview-B9h4wUga.js';
export { default as BaseExample } from './Example-CTYifxHH.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { U as UploadText } from './UploadText-BSD1IoCI.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './actions-Cit_UA5S.js';
import './DropdownArrow-Dw1NZKCM.js';
import './Square-B6INQV79.js';
import './Spinner-BrK-6emq.js';
import './StreamingBar-bVF1kkqe.js';
import './html-CdUEtR5E.js';
import './index-C7e-J7CF.js';
import './DownloadLink-KvrFKyMn.js';
import './Empty-CHa-gxv7.js';
import './ShareButton-Dz0y1C1K.js';
import './Download-pFhUaTl4.js';
import './IconButtonWrapper-Ba0GidgU.js';
import './Maximize-B_43ylTs.js';
import './VolumeLevels-DHAc0pUL.js';
import './Play-CWcFSzc7.js';
import './Undo-BxcJDM8E.js';
import './ModifyUpload-B5K-5Hqx.js';
import './Clear-Dgs5UhTO.js';
import './Edit-Dr58YXBX.js';
import './clone-DaB-S2nH.js';

var root_2$1 = from_html(`<div class="upload-container svelte-ey25pz"><!></div>`);
var root_12 = from_html(`<div class="file-name svelte-ey25pz"> </div> <div class="file-size svelte-ey25pz"> </div>`, 1);
var root_1 = from_html(`<!> <div data-testid="video" class="video-container svelte-ey25pz"><!> <!></div>`, 1);

function InteractiveVideo($$anchor, $$props) {
	push($$props, true);

	let value = prop($$props, 'value', 15, null),
		subtitle = prop($$props, 'subtitle', 3, null),
		sources = prop($$props, 'sources', 19, () => ["webcam", "upload"]),
		label = prop($$props, 'label', 3, undefined),
		show_download_button = prop($$props, 'show_download_button', 3, false),
		show_label = prop($$props, 'show_label', 3, true),
		initial_active_source = prop($$props, 'active_source', 3, "webcam"),
		handle_reset_value = prop($$props, 'handle_reset_value', 3, () => {}),
		max_file_size = prop($$props, 'max_file_size', 3, null),
		uploading = prop($$props, 'uploading', 15),
		upload_promise = prop($$props, 'upload_promise', 15),
		playback_position = prop($$props, 'playback_position', 15);
		prop($$props, 'buttons', 3, null);
		prop($$props, 'on_custom_button_click', 3, null);

	let has_change_history = state(false);

	let active_source = user_derived(() => {
		return initial_active_source() ?? "webcam";
	});

	function handle_load(detail) {
		value(detail);
		$$props.onchange?.(detail);

		if (detail) {
			$$props.onupload?.(detail);
		}
	}

	function handle_clear() {
		value(null);
		$$props.onchange?.(null);
		$$props.onclear?.();
	}

	function handle_change(video) {
		set(has_change_history, true);
		$$props.onchange?.(video);
	}

	function handle_capture({ detail }) {
		$$props.onchange?.(detail);
	}

	let dragging = state(false);

	user_effect(() => {
		$$props.ondrag?.(get(dragging));
	});

	var fragment = root_1();
	var node = first_child(fragment);

	{
		let $0 = user_derived(() => label() || "Video");

		BlockLabel(node, {
			get show_label() {
				return show_label();
			},

			get Icon() {
				return Video;
			},

			get label() {
				return get($0);
			}
		});
	}

	var div = sibling(node, 2);
	var node_1 = child(div);

	{
		var consequent_3 = ($$anchor) => {
			var div_1 = root_2$1();
			var node_2 = child(div_1);

			{
				var consequent_1 = ($$anchor) => {
					{
						let $0 = user_derived(() => $$props.i18n("video.drop_to_upload"));

						Upload($$anchor, {
							filetype: 'video/x-m4v,video/*',
							onload: handle_load,
							get max_file_size() {
								return max_file_size();
							},
							onerror: (detail) => $$props.onerror?.(detail),
							get root() {
								return $$props.root;
							},

							get upload() {
								return $$props.upload;
							},

							get stream_handler() {
								return $$props.stream_handler;
							},

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
								var node_3 = first_child(fragment_2);

								{
									var consequent = ($$anchor) => {
										var fragment_3 = comment();
										var node_4 = first_child(fragment_3);

										snippet(node_4, () => $$props.children);
										append($$anchor, fragment_3);
									};

									if_block(node_3, ($$render) => {
										if ($$props.children) $$render(consequent);
									});
								}

								append($$anchor, fragment_2);
							},
							$$slots: { default: true }
						});
					}
				};

				var alternate = ($$anchor) => {
					var fragment_4 = comment();
					var node_5 = first_child(fragment_4);

					{
						var consequent_2 = ($$anchor) => {
							Webcam($$anchor, {
								get root() {
									return $$props.root;
								},

								get mirror_webcam() {
									return $$props.webcam_options.mirror;
								},

								get webcam_constraints() {
									return $$props.webcam_options.constraints;
								},

								get include_audio() {
									return $$props.include_audio;
								},
								mode: 'video',
								get i18n() {
									return $$props.i18n;
								},

								get upload() {
									return $$props.upload;
								},
								stream_every: 1,
								$$events: {
									error: ({ detail }) => $$props.onerror?.(detail),
									capture: handle_capture,
									start_recording: () => $$props.onstart_recording?.(),
									stop_recording: () => $$props.onstop_recording?.()
								}
							});
						};

						if_block(
							node_5,
							($$render) => {
								if (get(active_source) === "webcam") $$render(consequent_2);
							},
							true
						);
					}

					append($$anchor, fragment_4);
				};

				if_block(node_2, ($$render) => {
					if (get(active_source) === "upload") $$render(consequent_1); else $$render(alternate, false);
				});
			}

			reset(div_1);
			append($$anchor, div_1);
		};

		var alternate_2 = ($$anchor) => {
			var fragment_6 = comment();
			var node_6 = first_child(fragment_6);

			{
				var consequent_4 = ($$anchor) => {
					var fragment_7 = comment();
					var node_7 = first_child(fragment_7);

					key(node_7, () => value()?.url, ($$anchor) => {
						{
							let $0 = user_derived(() => subtitle()?.url);
							let $1 = user_derived(() => $$props.webcam_options.mirror && get(active_source) === "webcam");

							Player($$anchor, {
								get upload() {
									return $$props.upload;
								},

								get root() {
									return $$props.root;
								},
								interactive: true,
								get autoplay() {
									return $$props.autoplay;
								},

								get src() {
									return value().url;
								},

								get subtitle() {
									return get($0);
								},
								is_stream: false,
								onplay: () => $$props.onplay?.(),
								onpause: () => $$props.onpause?.(),
								onstop: () => $$props.onstop?.(),
								onend: () => $$props.onend?.(),
								onerror: (error) => $$props.onerror?.(error),
								get mirror() {
									return get($1);
								},

								get label() {
									return label();
								},
								handle_change,
								get handle_reset_value() {
									return handle_reset_value();
								},

								get loop() {
									return $$props.loop;
								},

								get value() {
									return value();
								},

								get i18n() {
									return $$props.i18n;
								},

								get show_download_button() {
									return show_download_button();
								},
								handle_clear,
								get has_change_history() {
									return get(has_change_history);
								},

								get playback_position() {
									return playback_position();
								},

								set playback_position($$value) {
									playback_position($$value);
								}
							});
						}
					});

					append($$anchor, fragment_7);
				};

				var alternate_1 = ($$anchor) => {
					var fragment_9 = comment();
					var node_8 = first_child(fragment_9);

					{
						var consequent_5 = ($$anchor) => {
							var fragment_10 = root_12();
							var div_2 = first_child(fragment_10);
							var text = child(div_2, true);

							reset(div_2);

							var div_3 = sibling(div_2, 2);
							var text_1 = child(div_3, true);

							reset(div_3);

							template_effect(
								($0) => {
									set_text(text, value().orig_name || value().url);
									set_text(text_1, $0);
								},
								[() => prettyBytes(value().size)]
							);

							append($$anchor, fragment_10);
						};

						if_block(
							node_8,
							($$render) => {
								if (value().size) $$render(consequent_5);
							},
							true
						);
					}

					append($$anchor, fragment_9);
				};

				if_block(
					node_6,
					($$render) => {
						if (value()?.url) $$render(consequent_4); else $$render(alternate_1, false);
					},
					true
				);
			}

			append($$anchor, fragment_6);
		};

		if_block(node_1, ($$render) => {
			if (value() === null || value()?.url === undefined) $$render(consequent_3); else $$render(alternate_2, false);
		});
	}

	var node_9 = sibling(node_1, 2);

	SelectSource(node_9, {
		get sources() {
			return sources();
		},
		handle_clear,
		get active_source() {
			return get(active_source);
		},

		set active_source($$value) {
			set(active_source, $$value);
		}
	});

	reset(div);
	append($$anchor, fragment);
	pop();
}

var root_2 = from_html(`<!> <!>`, 1);
var root_4 = from_html(`<!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	let upload_promise = state(void 0);

	class VideoGradio extends Gradio {
		async get_data() {
			if (get(upload_promise)) {
				await get(upload_promise);
				await tick();
			}

			const data = await super.get_data();

			return data;
		}
	}

	const gradio = new VideoGradio(props);
	let old_value = gradio.props.value;
	let mounted = false;
	let uploading = state(false);
	let dragging = state(false);
	let active_source = user_derived(() => gradio.props.sources ? gradio.props.sources[0] : undefined);
	let initial_value = gradio.props.value;

	user_effect(() => {
		if (!mounted) {
			old_value = gradio.props.value;
			mounted = true;

			return;
		}

		if (old_value != gradio.props.value) {
			old_value = gradio.props.value;
			gradio.dispatch("change");
		}
	});

	const handle_reset_value = () => {
		if (initial_value === null || gradio.props.value === initial_value) {
			return;
		}

		gradio.props.value = initial_value;
	};

	function handle_change(detail) {
		if (detail != null) {
			gradio.props.value = detail;
		} else {
			gradio.props.value = null;
		}
	}

	function handle_error(detail) {
		const [level, status] = detail.includes("Invalid file type") ? ["warning", "complete"] : ["error", "error"];

		gradio.shared.loading_status.status = status;
		gradio.shared.loading_status.message = detail;
		gradio.dispatch(level, detail);
	}

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent = ($$anchor) => {
			{
				let $0 = user_derived(() => gradio.props.value === null && get(active_source) === "upload" ? "dashed" : "solid");
				let $1 = user_derived(() => get(dragging) ? "focus" : "base");
				let $2 = user_derived(() => gradio.props.height || undefined);

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

					get height() {
						return get($2);
					},

					get width() {
						return gradio.props.width;
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
							let $0 = user_derived(() => gradio.props.buttons ?? ["download", "share"]);

							VideoPreview(node_2, {
								get value() {
									return gradio.props.value;
								},

								get subtitle() {
									return gradio.props.subtitles;
								},

								get label() {
									return gradio.shared.label;
								},

								get show_label() {
									return gradio.shared.show_label;
								},

								get autoplay() {
									return gradio.props.autoplay;
								},

								get loop() {
									return gradio.props.loop;
								},

								get buttons() {
									return get($0);
								},

								on_custom_button_click: (id) => {
									gradio.dispatch("custom_button_click", { id });
								},
								onplay: () => gradio.dispatch("play"),
								onpause: () => gradio.dispatch("pause"),
								onstop: () => gradio.dispatch("stop"),
								onend: () => gradio.dispatch("end"),
								onshare: (detail) => gradio.dispatch("share", detail),
								onerror: (detail) => gradio.dispatch("error", detail),
								get i18n() {
									return gradio.i18n;
								},
								upload: (...args) => gradio.shared.client.upload(...args),
								get playback_position() {
									return gradio.props.playback_position;
								},

								set playback_position($$value) {
									gradio.props.playback_position = $$value;
								}
							});
						}

						append($$anchor, fragment_2);
					},
					$$slots: { default: true }
				});
			}
		};

		var alternate = ($$anchor) => {
			{
				let $0 = user_derived(() => gradio.props.value === null && get(active_source) === "upload" ? "dashed" : "solid");
				let $1 = user_derived(() => get(dragging) ? "focus" : "base");
				let $2 = user_derived(() => gradio.props.height || undefined);

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

					get height() {
						return get($2);
					},

					get width() {
						return gradio.props.width;
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
					children: ($$anchor, $$slotProps) => {
						var fragment_4 = root_4();
						var node_3 = first_child(fragment_4);

						Static(node_3, spread_props(
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

						var node_4 = sibling(node_3, 2);

						{
							let $0 = user_derived(() => gradio.props.buttons ?? ["download", "share"]);

							InteractiveVideo(node_4, {
								get value() {
									return gradio.props.value;
								},

								get subtitle() {
									return gradio.props.subtitles;
								},
								onchange: handle_change,
								ondrag: (detail) => set(dragging, detail, true),
								onerror: handle_error,
								get label() {
									return gradio.shared.label;
								},

								get show_label() {
									return gradio.shared.show_label;
								},

								get buttons() {
									return get($0);
								},

								on_custom_button_click: (id) => {
									gradio.dispatch("custom_button_click", { id });
								},

								get sources() {
									return gradio.props.sources;
								},

								get active_source() {
									return get(active_source);
								},

								get webcam_options() {
									return gradio.props.webcam_options;
								},

								get include_audio() {
									return gradio.props.include_audio;
								},

								get autoplay() {
									return gradio.props.autoplay;
								},

								get root() {
									return gradio.shared.root;
								},

								get loop() {
									return gradio.props.loop;
								},
								handle_reset_value,
								onclear: () => {
									gradio.props.value = null;
									gradio.dispatch("clear");
									gradio.dispatch("input");
								},
								onplay: () => gradio.dispatch("play"),
								onpause: () => gradio.dispatch("pause"),
								onupload: () => {
									gradio.dispatch("upload");
									gradio.dispatch("input");
								},
								onstop: () => gradio.dispatch("stop"),
								onend: () => gradio.dispatch("end"),
								onstart_recording: () => gradio.dispatch("start_recording"),
								onstop_recording: () => gradio.dispatch("stop_recording"),
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

								get uploading() {
									return get(uploading);
								},

								set uploading($$value) {
									set(uploading, $$value, true);
								},

								get playback_position() {
									return gradio.props.playback_position;
								},

								set playback_position($$value) {
									gradio.props.playback_position = $$value;
								},

								children: ($$anchor, $$slotProps) => {
									UploadText($$anchor, {
										get i18n() {
											return gradio.i18n;
										},
										type: 'video'
									});
								},
								$$slots: { default: true }
							});
						}

						append($$anchor, fragment_4);
					},
					$$slots: { default: true }
				});
			}
		};

		if_block(node, ($$render) => {
			if (!gradio.shared.interactive) $$render(consequent); else $$render(alternate, false);
		});
	}

	append($$anchor, fragment);
	pop();
}

export { InteractiveVideo as BaseInteractiveVideo, Player as BasePlayer, VideoPreview as BaseStaticVideo, Index as default, prettyBytes };
//# sourceMappingURL=index-DZgoAkOA.js.map
