import { p as prop, i as if_block, k as each, b as set_class, a as set_attribute, u as index, d as bind_this, f as set_style } from './i18n-CMo5lpzy.js';
import { av as FileData, R as push, u as state, v as proxy, y as user_effect, w as get, x as set, M as tick, aa as onMount, S as first_child, a as append, T as pop, a5 as user_derived, X as sibling, W as from_html, V as child, Y as reset, t as template_effect, Z as event, a0 as set_text } from './index-BvBk1Iap.js';
import { b as bind_window_size } from './window-BGXjFJtG.js';
import { s as stopPropagation } from './event-modifiers-DanhKw3_.js';
import { I as IconButton, U as Upload$1, W as Webcam, V as Video$1, a as ImagePaste } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { B as BlockLabel } from './BlockLabel-oGSqUii_.js';
import { E as Empty } from './Empty-CHa-gxv7.js';
import { S as ShareButton } from './ShareButton-Dz0y1C1K.js';
import { C as Clear } from './Clear-Dgs5UhTO.js';
import { D as Download } from './Download-pFhUaTl4.js';
import { I as Image } from './Image-C3F5KG1F.js';
import { P as Play } from './Play-CWcFSzc7.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { F as FullscreenButton } from './FullscreenButton-CNmhk6nd.js';
import { U as Upload } from './Upload-CPeZiqam.js';
import { M as ModifyUpload } from './ModifyUpload-B5K-5Hqx.js';
import { I as Image$1 } from './Image-CDw5qb5O.js';
/* empty css                                                    */
import { V as Video } from './Video-B3xDM6GN.js';
import { d as dequal } from './index-DnoGeqVF.js';
import { u as uploadToHuggingFace } from './utils.svelte-CYLB_jIA.js';

async function format_gallery_for_sharing(value) {
  if (!value) return "";
  let urls = await Promise.all(
    value.map(async ([image, _]) => {
      if (image === null || !image.url) return "";
      return await uploadToHuggingFace(image.url);
    })
  );
  return `<div style="display: flex; flex-wrap: wrap; gap: 16px">${urls.map((url) => `<img src="${url}" style="height: 400px" />`).join("")}</div>`;
}
async function handle_save(img_blob, upload, filename = "uploaded_file") {
  const ext = img_blob.type.split("/")[1] || "png";
  const f_ = new File([img_blob], `${filename}.${ext}`);
  const files = [
    new FileData({
      path: f_.name,
      orig_name: f_.name,
      blob: f_,
      size: f_.size,
      mime_type: f_.type,
      is_stream: false
    })
  ];
  return await upload(...files);
}

var root_10 = from_html(`<div class="icon-button"><!></div>`);
var root_7 = from_html(`<!> <!> <!> <!>`, 1);
var root_14 = from_html(`<caption class="caption svelte-7anmrz"> </caption>`);
var root_17 = from_html(`<!> <!>`, 1);
var root_15 = from_html(`<button><!></button>`);
var root_6 = from_html(`<span><!> <button class="media-button svelte-7anmrz" aria-label="detailed view of selected image"><!></button> <!> <div class="thumbnails scroll-hide svelte-7anmrz" data-testid="container_el"></div></span>`);
var root_19 = from_html(`<!> <!> <!> <!>`, 1);
var root_27 = from_html(`<!> <!>`, 1);
var root_28 = from_html(`<div class="caption-label svelte-7anmrz"> </div>`);
var root_29 = from_html(`<button class="delete-button svelte-7anmrz" aria-label="Delete image"><!></button>`);
var root_25 = from_html(`<div class="gallery-item svelte-7anmrz"><button><!> <!></button> <!></div>`);
var root_5 = from_html(`<div class="gallery-container svelte-7anmrz"><!> <div><!> <div></div></div></div>`);
var root_1 = from_html(`<!> <!>`, 1);

function Gallery($$anchor, $$props) {
	push($$props, true);

	let show_label = prop($$props, 'show_label', 3, true),
		value = prop($$props, 'value', 15),
		columns = prop($$props, 'columns', 19, () => [2]),
		rows = prop($$props, 'rows', 3, undefined),
		height = prop($$props, 'height', 3, "auto"),
		allow_preview = prop($$props, 'allow_preview', 3, true),
		object_fit = prop($$props, 'object_fit', 3, "cover"),
		show_share_button = prop($$props, 'show_share_button', 3, false),
		show_download_button = prop($$props, 'show_download_button', 3, false),
		selected_index = prop($$props, 'selected_index', 15),
		show_fullscreen_button = prop($$props, 'show_fullscreen_button', 3, true),
		fullscreen = prop($$props, 'fullscreen', 3, false),
		root = prop($$props, 'root', 3, ""),
		file_types = prop($$props, 'file_types', 19, () => ["image", "video"]),
		max_file_size = prop($$props, 'max_file_size', 3, null),
		fit_columns = prop($$props, 'fit_columns', 3, true),
		buttons = prop($$props, 'buttons', 3, null),
		oncustom_button_click = prop($$props, 'oncustom_button_click', 3, null),
		onpreview_open = prop($$props, 'onpreview_open', 3, () => {}),
		onchange = prop($$props, 'onchange', 3, () => {}),
		onclear = prop($$props, 'onclear', 3, () => {}),
		onselect = prop($$props, 'onselect', 3, (data) => {}),
		onshare = prop($$props, 'onshare', 3, (data) => {}),
		onerror = prop($$props, 'onerror', 3, (error) => {}),
		onpreview_close = prop($$props, 'onpreview_close', 3, () => {}),
		onfullscreen = prop($$props, 'onfullscreen', 3, (data) => {}),
		ondelete = prop($$props, 'ondelete', 3, () => {}),
		onupload = prop($$props, 'onupload', 3, () => {}),
		onsource_change = prop($$props, 'onsource_change', 3, () => {});

	let upload_promise = null;
	let mode = "normal";
	let display_icon_button_wrapper_top_corner = false;
	let is_full_screen = false;
	let image_container;
	let was_reset = state(false);

	let resolved_value = user_derived(() => value() == null
		? null
		: value().map((data) => {
			if ("video" in data) {
				return { video: data.video, caption: data.caption };
			} else if ("image" in data) {
				return { image: data.image, caption: data.caption };
			}

			return {};
		}));

	function resolve_effective_columns(resolved_value, columns, fit_columns) {
		if (resolved_value && columns && fit_columns) {
			const item_count = resolved_value.length;

			if (Array.isArray(columns)) {
				return columns.map((col) => Math.min(col, item_count));
			} else {
				return Math.min(columns, item_count);
			}
		} else {
			return columns;
		}
	}

	let effective_columns = user_derived(() => resolve_effective_columns(get(resolved_value), columns(), fit_columns()));
	let prev_value = state(proxy(value()));

	if (selected_index() == null && $$props.preview && value()?.length) {
		selected_index(0);
	}

	let old_selected_index = state(proxy(selected_index()));

	user_effect(() => {
		if (!dequal(get(prev_value), value())) {
			// When value is falsy (clear button or first load),
			// preview determines the selected image
			if (get(was_reset)) {
				selected_index($$props.preview && value()?.length ? 0 : null);
				set(was_reset, false);

				// Otherwise we keep the selected_index the same if the
				// gallery has at least as many elements as it did before
			} else {
				if (selected_index() !== null && value() !== null) {
					selected_index(Math.max(0, Math.min(selected_index(), value().length - 1)));
				} else {
					selected_index(null);
				}
			}

			onchange()();
			set(prev_value, value(), true);
		}
	});

	let previous = user_derived(() => ((selected_index() ?? 0) + (get(resolved_value)?.length ?? 0) - 1) % (get(resolved_value)?.length ?? 0));
	let next = user_derived(() => ((selected_index() ?? 0) + 1) % (get(resolved_value)?.length ?? 0));

	function handle_preview_click(event) {
		const element = event.target;
		const x = event.offsetX;
		const width = element.offsetWidth;
		const centerX = width / 2;

		if (x < centerX) {
			selected_index(get(previous));
		} else {
			selected_index(get(next));
		}
	}

	function on_keydown(e) {
		switch (e.code) {
			case "Escape":
				e.preventDefault();
				selected_index(null);
				onpreview_close()();
				break;

			case "ArrowLeft":
				e.preventDefault();
				selected_index(get(previous));
				break;

			case "ArrowRight":
				e.preventDefault();
				selected_index(get(next));
				break;
		}
	}

	user_effect(() => {
		// Only trigger the effect if the selected_index has actually changed
		if (selected_index() !== get(old_selected_index)) {
			set(old_selected_index, selected_index(), true);

			// Ensure we have a valid selection and data to work with
			if (selected_index() !== null && get(resolved_value) !== null) {
				// Notify the parent component or listener about the updated selection
				onselect()({
					index: selected_index(),
					value: get(resolved_value)?.[selected_index()]
				});
			}
		}
	});

	user_effect(() => {
		if (allow_preview() && get(container_element)) {
			scroll_to_img(selected_index());
		}
	});

	let el = [];
	let container_element = state(void 0);

	async function scroll_to_img(index) {
		if (typeof index !== "number") return;

		await tick();

		if (el[index] === undefined) return;
		if (!get(container_element)) return;

		el[index]?.focus();

		const { left: container_left, width: container_width } = get(container_element).getBoundingClientRect();
		const { left, width } = el[index].getBoundingClientRect();
		const relative_left = left - container_left;
		const pos = relative_left + width / 2 - container_width / 2 + get(container_element).scrollLeft;

		if (get(container_element) && typeof get(container_element).scrollTo === "function") {
			get(container_element).scrollTo({ left: pos < 0 ? 0 : pos, behavior: "smooth" });
		}
	}

	// Unlike `gr.Image()`, images specified via remote URLs are not cached in the server
	// and their remote URLs are directly passed to the client as `value[].image.url`.
	// The `download` attribute of the <a> tag doesn't work for remote URLs (https://developer.mozilla.org/en-US/docs/Web/HTML/Element/a#download),
	// so we need to download the image via JS as below.
	async function download(file_url, name) {
		let response;

		try {
			response = await $$props._fetch(file_url);
		} catch(error) {
			if (error instanceof TypeError) {
				// If CORS is not allowed (https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch#checking_that_the_fetch_was_successful),
				// open the link in a new tab instead, mimicing the behavior of the `download` attribute for remote URLs,
				// which is not ideal, but a reasonable fallback.
				window.open(file_url, "_blank", "noreferrer");

				return;
			}

			throw error;
		}

		const blob = await response.blob();
		const url = URL.createObjectURL(blob);
		const link = document.createElement("a");

		link.href = url;
		link.download = name;
		link.click();
		URL.revokeObjectURL(url);
	}

	let selected_media = user_derived(() => selected_index() != null && get(resolved_value) != null ? get(resolved_value)[selected_index()] : null);
	let thumbnails_overflow = false;

	function check_thumbnails_overflow() {
		if (get(container_element)) {
			thumbnails_overflow = get(container_element).scrollWidth > get(container_element).clientWidth;
		}
	}

	onMount(() => {
		check_thumbnails_overflow();

		document.addEventListener("fullscreenchange", () => {
			is_full_screen = !!document.fullscreenElement;
		});

		window.addEventListener("resize", check_thumbnails_overflow);

		return () => window.removeEventListener("resize", check_thumbnails_overflow);
	});

	user_effect(() => {
		get(resolved_value);
		check_thumbnails_overflow();

		if (get(container_element)) {
			check_thumbnails_overflow();
		}
	});

	function handle_item_delete(index) {
		if (!value() || !get(resolved_value)) return;

		const deleted_item = get(resolved_value)[index];
		let deleted_file_data;

		if ("image" in deleted_item) {
			deleted_file_data = { file: deleted_item.image, index };
		} else if ("video" in deleted_item) {
			deleted_file_data = { file: deleted_item.video, index };
		}

		if (deleted_file_data) {
			ondelete()(deleted_file_data);
		}
	}

	let uploading = false;
	var fragment = root_1();
	var node = first_child(fragment);

	{
		var consequent = ($$anchor) => {
			{
				let $0 = user_derived(() => $$props.label || "Gallery");

				BlockLabel($$anchor, {
					get show_label() {
						return show_label();
					},

					get Icon() {
						return Image;
					},

					get label() {
						return get($0);
					}
				});
			}
		};

		if_block(node, ($$render) => {
			if (show_label()) $$render(consequent);
		});
	}

	var node_1 = sibling(node, 2);

	{
		var consequent_1 = ($$anchor) => {
			Empty($$anchor, {
				unpadded_box: true,
				size: 'large',
				children: ($$anchor, $$slotProps) => {
					Image($$anchor);
				},
				$$slots: { default: true }
			});
		};

		var alternate_3 = ($$anchor) => {
			var div = root_5();
			var node_2 = child(div);

			{
				var consequent_9 = ($$anchor) => {
					var span = root_6();

					set_class(span, 1, 'preview svelte-7anmrz', null, {}, { minimal: mode === "minimal" });

					var node_3 = child(span);

					IconButtonWrapper(node_3, {
						display_top_corner: display_icon_button_wrapper_top_corner,
						get buttons() {
							return buttons();
						},

						get on_custom_button_click() {
							return oncustom_button_click();
						},

						children: ($$anchor, $$slotProps) => {
							var fragment_4 = root_7();
							var node_4 = first_child(fragment_4);

							{
								var consequent_2 = ($$anchor) => {
									{
										let $0 = user_derived(() => $$props.i18n("common.download"));

										IconButton($$anchor, {
											get Icon() {
												return Download;
											},

											get label() {
												return get($0);
											},

											onclick: () => {
												const image = "image" in get(selected_media)
													? get(selected_media)?.image
													: get(selected_media)?.video;

												if (image == null) {
													return;
												}

												const { url, orig_name } = image;

												if (url) {
													download(url, orig_name ?? "image");
												}
											}
										});
									}
								};

								if_block(node_4, ($$render) => {
									if (show_download_button()) $$render(consequent_2);
								});
							}

							var node_5 = sibling(node_4, 2);

							{
								var consequent_3 = ($$anchor) => {
									FullscreenButton($$anchor, {
										get fullscreen() {
											return fullscreen();
										},
										onclick: (is_fullscreen) => onfullscreen()({ detail: is_fullscreen })
									});
								};

								if_block(node_5, ($$render) => {
									if (show_fullscreen_button()) $$render(consequent_3);
								});
							}

							var node_6 = sibling(node_5, 2);

							{
								var consequent_4 = ($$anchor) => {
									var div_1 = root_10();
									var node_7 = child(div_1);

									ShareButton(node_7, {
										get i18n() {
											return $$props.i18n;
										},

										get value() {
											return get(resolved_value);
										},

										get formatter() {
											return format_gallery_for_sharing;
										},

										$$events: {
											share: (detail) => {
												onshare()(detail);
											},

											error: (detail) => {
												onerror()(detail);
											}
										}
									});

									reset(div_1);
									append($$anchor, div_1);
								};

								if_block(node_6, ($$render) => {
									if (show_share_button()) $$render(consequent_4);
								});
							}

							var node_8 = sibling(node_6, 2);

							{
								var consequent_5 = ($$anchor) => {
									IconButton($$anchor, {
										get Icon() {
											return Clear;
										},
										label: 'Close',
										onclick: () => {
											selected_index(null);
											onpreview_close()();
										}
									});
								};

								if_block(node_8, ($$render) => {
									if (!is_full_screen) $$render(consequent_5);
								});
							}

							append($$anchor, fragment_4);
						},
						$$slots: { default: true }
					});

					var button = sibling(node_3, 2);
					var node_9 = child(button);

					{
						var consequent_6 = ($$anchor) => {
							{
								let $0 = user_derived(() => ({
									alt: get(selected_media).caption || "",
									title: get(selected_media).caption || null,
									class: get(selected_media).caption && "with-caption",
									loading: "lazy"
								}));

								Image$1($$anchor, {
									get restProps() {
										return get($0);
									},

									get src() {
										return get(selected_media).image.url;
									},
									data_testid: 'detailed-image'
								});
							}
						};

						var alternate = ($$anchor) => {
							{
								let $0 = user_derived(() => get(selected_media).caption || "");

								Video($$anchor, {
									get src() {
										return get(selected_media).video.url;
									},
									'data-testid': "detailed-video",
									get alt() {
										return get($0);
									},
									loading: 'lazy',
									loop: false,
									is_stream: false,
									muted: false,
									controls: true
								});
							}
						};

						if_block(node_9, ($$render) => {
							if ("image" in get(selected_media)) $$render(consequent_6); else $$render(alternate, false);
						});
					}

					reset(button);

					var node_10 = sibling(button, 2);

					{
						var consequent_7 = ($$anchor) => {
							var caption = root_14();
							var text = child(caption, true);

							reset(caption);
							template_effect(() => set_text(text, get(selected_media).caption));
							append($$anchor, caption);
						};

						if_block(node_10, ($$render) => {
							if (get(selected_media)?.caption) $$render(consequent_7);
						});
					}

					var div_2 = sibling(node_10, 2);

					each(div_2, 21, () => get(resolved_value), index, ($$anchor, media, i) => {
						var button_1 = root_15();
						let classes;
						var node_11 = child(button_1);

						{
							var consequent_8 = ($$anchor) => {
								{
									let $0 = user_derived(() => ({
										title: get(media).caption || null,
										alt: "",
										class: "with-caption",
										loading: "lazy"
									}));

									Image$1($$anchor, {
										get src() {
											return get(media).image.url;
										},

										get restProps() {
											return get($0);
										},
										data_testid: `thumbnail ${i + 1}`
									});
								}
							};

							var alternate_1 = ($$anchor) => {
								var fragment_11 = root_17();
								var node_12 = first_child(fragment_11);

								Play(node_12);

								var node_13 = sibling(node_12, 2);

								{
									let $0 = user_derived(() => get(media).caption || null);

									Video(node_13, {
										get src() {
											return get(media).video.url;
										},

										get title() {
											return get($0);
										},
										is_stream: false,
										'data-testid': "thumbnail " + (i + 1),
										alt: '',
										loading: 'lazy',
										loop: false
									});
								}

								append($$anchor, fragment_11);
							};

							if_block(node_11, ($$render) => {
								if ("image" in get(media)) $$render(consequent_8); else $$render(alternate_1, false);
							});
						}

						reset(button_1);
						bind_this(button_1, ($$value, i) => el[i] = $$value, (i) => el?.[i], () => [i]);

						template_effect(() => {
							classes = set_class(button_1, 1, 'thumbnail-item thumbnail-small svelte-7anmrz', null, classes, { selected: selected_index() === i && mode !== "minimal" });
							set_attribute(button_1, 'aria-label', "Thumbnail " + (i + 1) + " of " + get(resolved_value).length);
						});

						event('click', button_1, () => selected_index(i));
						append($$anchor, button_1);
					});

					reset(div_2);
					bind_this(div_2, ($$value) => set(container_element, $$value), () => get(container_element));
					reset(span);

					template_effect(() => {
						set_style(button, `height: calc(100% - ${get(selected_media).caption ? '80px' : '60px'})`);
						set_style(div_2, `justify-content: ${thumbnails_overflow ? 'flex-start' : 'center'};`);
					});

					event('click', button, function (...$$args) {
						("image" in get(selected_media) ? (event) => handle_preview_click(event) : null)?.apply(this, $$args);
					});

					event('keydown', span, on_keydown);
					append($$anchor, span);
				};

				if_block(node_2, ($$render) => {
					if (get(selected_media) && allow_preview()) $$render(consequent_9);
				});
			}

			var div_3 = sibling(node_2, 2);
			let classes_1;
			let styles;
			var node_14 = child(div_3);

			{
				var consequent_14 = ($$anchor) => {
					ModifyUpload($$anchor, {
						get i18n() {
							return $$props.i18n;
						},

						onclear: () => {
							value([]);
							onsource_change()("upload");
							onclear()();
						},

						children: ($$anchor, $$slotProps) => {
							var fragment_13 = root_19();
							var node_15 = first_child(fragment_13);

							{
								var consequent_10 = ($$anchor) => {
									{
										let $0 = user_derived(() => $$props.i18n("upload_text.click_to_upload"));

										IconButton($$anchor, {
											get Icon() {
												return Upload$1;
											},

											get label() {
												return get($0);
											},

											children: ($$anchor, $$slotProps) => {
												Upload($$anchor, {
													icon_upload: true,
													onload: (e) => onupload()(e),
													get filetype() {
														return file_types();
													},
													file_count: 'multiple',
													get max_file_size() {
														return max_file_size();
													},

													get root() {
														return root();
													},
													onerror: (e) => onerror()(e),
													get stream_handler() {
														return $$props.stream_handler;
													},

													get upload() {
														return $$props.upload;
													},

													get upload_promise() {
														return upload_promise;
													},

													set upload_promise($$value) {
														upload_promise = $$value;
													},

													get uploading() {
														return uploading;
													},

													set uploading($$value) {
														uploading = $$value;
													}
												});
											},
											$$slots: { default: true }
										});
									}
								};

								if_block(node_15, ($$render) => {
									if ($$props.upload && $$props.stream_handler) $$render(consequent_10);
								});
							}

							var node_16 = sibling(node_15, 2);

							{
								var consequent_11 = ($$anchor) => {
									{
										let $0 = user_derived(() => $$props.i18n("common.webcam"));

										IconButton($$anchor, {
											get Icon() {
												return Webcam;
											},

											get label() {
												return get($0);
											},

											onclick: () => {
												onsource_change()("webcam");
											}
										});
									}
								};

								if_block(node_16, ($$render) => {
									if ($$props.sources.includes("webcam")) $$render(consequent_11);
								});
							}

							var node_17 = sibling(node_16, 2);

							{
								var consequent_12 = ($$anchor) => {
									{
										let $0 = user_derived(() => $$props.i18n("common.video"));

										IconButton($$anchor, {
											get Icon() {
												return Video$1;
											},

											get label() {
												return get($0);
											},

											onclick: () => {
												onsource_change()("webcam-video");
											}
										});
									}
								};

								if_block(node_17, ($$render) => {
									if ($$props.sources.includes("webcam-video")) $$render(consequent_12);
								});
							}

							var node_18 = sibling(node_17, 2);

							{
								var consequent_13 = ($$anchor) => {
									{
										let $0 = user_derived(() => $$props.i18n("upload_text.paste_clipboard"));

										IconButton($$anchor, {
											get label() {
												return get($0);
											},

											onclick: () => {
												onsource_change()("clipboard");
											},

											get Icon() {
												return ImagePaste;
											}
										});
									}
								};

								if_block(node_18, ($$render) => {
									if ($$props.sources.includes("clipboard")) $$render(consequent_13);
								});
							}

							append($$anchor, fragment_13);
						},
						$$slots: { default: true }
					});
				};

				if_block(node_14, ($$render) => {
					if ($$props.interactive && selected_index() === null) $$render(consequent_14);
				});
			}

			var div_4 = sibling(node_14, 2);
			let classes_2;

			each(div_4, 21, () => get(resolved_value), index, ($$anchor, entry, i) => {
				var div_5 = root_25();
				var button_2 = child(div_5);
				let classes_3;
				var node_19 = child(button_2);

				{
					var consequent_15 = ($$anchor) => {
						{
							let $0 = user_derived(() => get(entry).caption || "");
							let $1 = user_derived(() => typeof get(entry).image === "string" ? get(entry).image : get(entry).image.url);

							Image$1($$anchor, {
								get alt() {
									return get($0);
								},

								get src() {
									return get($1);
								},
								loading: 'lazy'
							});
						}
					};

					var alternate_2 = ($$anchor) => {
						var fragment_20 = root_27();
						var node_20 = first_child(fragment_20);

						Play(node_20);

						var node_21 = sibling(node_20, 2);

						{
							let $0 = user_derived(() => get(entry).caption || null);

							Video(node_21, {
								get src() {
									return get(entry).video.url;
								},

								get title() {
									return get($0);
								},
								is_stream: false,
								'data-testid': "thumbnail " + (i + 1),
								alt: '',
								loading: 'lazy',
								loop: false
							});
						}

						append($$anchor, fragment_20);
					};

					if_block(node_19, ($$render) => {
						if ("image" in get(entry)) $$render(consequent_15); else $$render(alternate_2, false);
					});
				}

				var node_22 = sibling(node_19, 2);

				{
					var consequent_16 = ($$anchor) => {
						var div_6 = root_28();
						var text_1 = child(div_6, true);

						reset(div_6);
						template_effect(() => set_text(text_1, get(entry).caption));
						append($$anchor, div_6);
					};

					if_block(node_22, ($$render) => {
						if (get(entry).caption) $$render(consequent_16);
					});
				}

				reset(button_2);

				var node_23 = sibling(button_2, 2);

				{
					var consequent_17 = ($$anchor) => {
						var button_3 = root_29();
						var node_24 = child(button_3);

						Clear(node_24);
						reset(button_3);
						event('click', button_3, stopPropagation(() => handle_item_delete(i)));
						append($$anchor, button_3);
					};

					if_block(node_23, ($$render) => {
						if ($$props.interactive) $$render(consequent_17);
					});
				}

				reset(div_5);

				template_effect(() => {
					classes_3 = set_class(button_2, 1, 'thumbnail-item thumbnail-lg svelte-7anmrz', null, classes_3, { selected: selected_index() === i });
					set_attribute(button_2, 'aria-label', "Thumbnail " + (i + 1) + " of " + get(resolved_value).length);
				});

				event('click', button_2, () => {
					if (selected_index() === null && allow_preview()) {
						onpreview_open()();
					}

					selected_index(i);
				});

				append($$anchor, div_5);
			});

			reset(div_4);
			reset(div_3);
			reset(div);
			bind_this(div, ($$value) => image_container = $$value, () => image_container);

			template_effect(() => {
				classes_1 = set_class(div_3, 1, 'grid-wrap svelte-7anmrz', null, classes_1, {
					minimal: mode === "minimal",
					'fixed-height': (!height() || height() == "auto"),
					hidden: is_full_screen
				});

				styles = set_style(div_3, '', styles, {
					height: height() !== "auto"
						? typeof height() === "number" ? height() + "px" : height()
						: null
				});

				classes_2 = set_class(div_4, 1, 'grid-container svelte-7anmrz', null, classes_2, { 'pt-6': show_label() });
				set_style(div_4, `--grid-cols:${get(effective_columns) ?? ''}; --grid-rows:${rows() ?? ''}; --object-fit: ${object_fit() ?? ''};`);
			});

			append($$anchor, div);
		};

		if_block(node_1, ($$render) => {
			if (value() == null || get(resolved_value) == null || get(resolved_value).length === 0) $$render(consequent_1); else $$render(alternate_3, false);
		});
	}

	bind_window_size('innerHeight', ($$value) => $$value);
	append($$anchor, fragment);
	pop();
}

const Gallery$1 = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
	__proto__: null,
	default: Gallery
}, Symbol.toStringTag, { value: 'Module' }));

export { Gallery as G, Gallery$1 as a, handle_save as h };
//# sourceMappingURL=Gallery-DaBd1NXs.js.map
