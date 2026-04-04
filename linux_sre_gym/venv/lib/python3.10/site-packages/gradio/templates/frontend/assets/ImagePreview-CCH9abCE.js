import { p as prop, e as init, i as if_block, d as bind_this, b as set_class } from './i18n-CMo5lpzy.js';
import { R as push, a4 as createEventDispatcher, S as first_child, w as get, a as append, T as pop, U as flushSync, ad as derived_safe_equal, G as deep_read_state, z as untrack, X as sibling, W as from_html, V as child, Y as reset, x as set, t as template_effect, Z as event, a3 as mutable_source } from './index-BvBk1Iap.js';
import { b as bubble_event } from './misc-ByVveAau.js';
import { u as uploadToHuggingFace } from './utils.svelte-CYLB_jIA.js';
import { I as IconButton } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { B as BlockLabel } from './BlockLabel-oGSqUii_.js';
import { D as DownloadLink } from './DownloadLink-KvrFKyMn.js';
import { E as Empty } from './Empty-CHa-gxv7.js';
import { S as ShareButton } from './ShareButton-Dz0y1C1K.js';
import { D as Download } from './Download-pFhUaTl4.js';
import { I as Image } from './Image-C3F5KG1F.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { F as FullscreenButton } from './FullscreenButton-CNmhk6nd.js';
import { I as Image$1 } from './Image-CDw5qb5O.js';
/* empty css                                                    */

const get_coordinates_of_clicked_image = (evt) => {
  let image;
  if (evt.currentTarget instanceof Element) {
    image = evt.currentTarget.querySelector("img");
  } else {
    return [NaN, NaN];
  }
  const imageRect = image.getBoundingClientRect();
  const xScale = image.naturalWidth / imageRect.width;
  const yScale = image.naturalHeight / imageRect.height;
  if (xScale > yScale) {
    const displayed_height = image.naturalHeight / xScale;
    const y_offset = (imageRect.height - displayed_height) / 2;
    var x = Math.round((evt.clientX - imageRect.left) * xScale);
    var y = Math.round((evt.clientY - imageRect.top - y_offset) * xScale);
  } else {
    const displayed_width = image.naturalWidth / yScale;
    const x_offset = (imageRect.width - displayed_width) / 2;
    var x = Math.round((evt.clientX - imageRect.left - x_offset) * yScale);
    var y = Math.round((evt.clientY - imageRect.top) * yScale);
  }
  if (x < 0 || x >= image.naturalWidth || y < 0 || y >= image.naturalHeight) {
    return null;
  }
  return [x, y];
};

var root_4 = from_html(`<!> <!> <!>`, 1);
var root_3 = from_html(`<div class="image-container svelte-12vrxzd"><!> <button class="svelte-12vrxzd"><div><!></div></button></div>`);
var root = from_html(`<!> <!>`, 1);

function ImagePreview($$anchor, $$props) {
	push($$props, false);

	let value = prop($$props, 'value', 12);
	let label = prop($$props, 'label', 12, undefined);
	let show_label = prop($$props, 'show_label', 12);
	let buttons = prop($$props, 'buttons', 28, () => []);
	let on_custom_button_click = prop($$props, 'on_custom_button_click', 12, null);
	let selectable = prop($$props, 'selectable', 12, false);
	let i18n = prop($$props, 'i18n', 12);
	let display_icon_button_wrapper_top_corner = prop($$props, 'display_icon_button_wrapper_top_corner', 12, false);
	let fullscreen = prop($$props, 'fullscreen', 12, false);
	let show_button_background = prop($$props, 'show_button_background', 12, true);
	const dispatch = createEventDispatcher();

	const handle_click = (evt) => {
		let coordinates = get_coordinates_of_clicked_image(evt);

		if (coordinates) {
			dispatch("select", { index: coordinates, value: null });
		}
	};

	let image_container = mutable_source();

	var $$exports = {
		get value() {
			return value();
		},

		set value($$value) {
			value($$value);
			flushSync();
		},

		get label() {
			return label();
		},

		set label($$value) {
			label($$value);
			flushSync();
		},

		get show_label() {
			return show_label();
		},

		set show_label($$value) {
			show_label($$value);
			flushSync();
		},

		get buttons() {
			return buttons();
		},

		set buttons($$value) {
			buttons($$value);
			flushSync();
		},

		get on_custom_button_click() {
			return on_custom_button_click();
		},

		set on_custom_button_click($$value) {
			on_custom_button_click($$value);
			flushSync();
		},

		get selectable() {
			return selectable();
		},

		set selectable($$value) {
			selectable($$value);
			flushSync();
		},

		get i18n() {
			return i18n();
		},

		set i18n($$value) {
			i18n($$value);
			flushSync();
		},

		get display_icon_button_wrapper_top_corner() {
			return display_icon_button_wrapper_top_corner();
		},

		set display_icon_button_wrapper_top_corner($$value) {
			display_icon_button_wrapper_top_corner($$value);
			flushSync();
		},

		get fullscreen() {
			return fullscreen();
		},

		set fullscreen($$value) {
			fullscreen($$value);
			flushSync();
		},

		get show_button_background() {
			return show_button_background();
		},

		set show_button_background($$value) {
			show_button_background($$value);
			flushSync();
		}
	};

	init();

	var fragment = root();
	var node = first_child(fragment);

	{
		let $0 = derived_safe_equal(() => (
			deep_read_state(show_label()),
			deep_read_state(label()),
			deep_read_state(i18n()),
			untrack(() => !show_label() ? "" : label() || i18n()("image.image"))
		));

		BlockLabel(node, {
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

	var node_1 = sibling(node, 2);

	{
		var consequent = ($$anchor) => {
			Empty($$anchor, {
				unpadded_box: true,
				size: 'large',
				children: ($$anchor, $$slotProps) => {
					Image($$anchor);
				},
				$$slots: { default: true }
			});
		};

		var alternate = ($$anchor) => {
			var div = root_3();
			var node_2 = child(div);

			IconButtonWrapper(node_2, {
				get display_top_corner() {
					return display_icon_button_wrapper_top_corner();
				},

				get show_background() {
					return show_button_background();
				},

				get buttons() {
					return buttons();
				},

				get on_custom_button_click() {
					return on_custom_button_click();
				},

				children: ($$anchor, $$slotProps) => {
					var fragment_3 = root_4();
					var node_3 = first_child(fragment_3);

					{
						var consequent_1 = ($$anchor) => {
							FullscreenButton($$anchor, {
								get fullscreen() {
									return fullscreen();
								},

								onclick: (is_fullscreen) => {
									fullscreen(is_fullscreen);
									dispatch("fullscreen", is_fullscreen);
								}
							});
						};

						if_block(node_3, ($$render) => {
							if ((
								deep_read_state(buttons()),
								untrack(() => buttons().some((btn) => typeof btn === "string" && btn === "fullscreen"))
							)) $$render(consequent_1);
						});
					}

					var node_4 = sibling(node_3, 2);

					{
						var consequent_2 = ($$anchor) => {
							{
								let $0 = derived_safe_equal(() => (
									deep_read_state(value()),
									untrack(() => value().orig_name || "image")
								));

								DownloadLink($$anchor, {
									get href() {
										return (deep_read_state(value()), untrack(() => value().url));
									},

									get download() {
										return get($0);
									},

									children: ($$anchor, $$slotProps) => {
										{
											let $0 = derived_safe_equal(() => (
												deep_read_state(i18n()),
												untrack(() => i18n()("common.download"))
											));

											IconButton($$anchor, {
												get Icon() {
													return Download;
												},

												get label() {
													return get($0);
												}
											});
										}
									},
									$$slots: { default: true }
								});
							}
						};

						if_block(node_4, ($$render) => {
							if ((
								deep_read_state(buttons()),
								untrack(() => buttons().some((btn) => typeof btn === "string" && btn === "download"))
							)) $$render(consequent_2);
						});
					}

					var node_5 = sibling(node_4, 2);

					{
						var consequent_3 = ($$anchor) => {
							ShareButton($$anchor, {
								get i18n() {
									return i18n();
								},

								formatter: async (value) => {
									if (!value) return "";

									let url = await uploadToHuggingFace(value);

									return `<img src="${url}" />`;
								},

								get value() {
									return value();
								},

								$$events: {
									share($$arg) {
										bubble_event.call(this, $$props, $$arg);
									},

									error($$arg) {
										bubble_event.call(this, $$props, $$arg);
									}
								}
							});
						};

						if_block(node_5, ($$render) => {
							if ((
								deep_read_state(buttons()),
								untrack(() => buttons().some((btn) => typeof btn === "string" && btn === "share"))
							)) $$render(consequent_3);
						});
					}

					append($$anchor, fragment_3);
				},
				$$slots: { default: true }
			});

			var button = sibling(node_2, 2);
			var div_1 = child(button);
			let classes;
			var node_6 = child(div_1);

			Image$1(node_6, {
				get src() {
					return (deep_read_state(value()), untrack(() => value().url));
				},
				restProps: { loading: "lazy", alt: "" },
				$$events: {
					load($$arg) {
						bubble_event.call(this, $$props, $$arg);
					}
				}
			});

			reset(div_1);
			reset(button);
			reset(div);
			bind_this(div, ($$value) => set(image_container, $$value), () => get(image_container));
			template_effect(() => classes = set_class(div_1, 1, 'image-frame svelte-12vrxzd', null, classes, { selectable: selectable() }));
			event('click', button, handle_click);
			append($$anchor, div);
		};

		if_block(node_1, ($$render) => {
			if ((
				deep_read_state(value()),
				untrack(() => value() == null || !value()?.url)
			)) $$render(consequent); else $$render(alternate, false);
		});
	}

	append($$anchor, fragment);

	return pop($$exports);
}

const ImagePreview$1 = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
	__proto__: null,
	default: ImagePreview
}, Symbol.toStringTag, { value: 'Module' }));

export { ImagePreview as I, ImagePreview$1 as a, get_coordinates_of_clicked_image as g };
//# sourceMappingURL=ImagePreview-CCH9abCE.js.map
