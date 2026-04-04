import { p as prop, i as if_block, a as set_attribute, r as rest_props } from './i18n-CMo5lpzy.js';
import { R as push, S as first_child, a as append, T as pop, X as sibling, W as from_html, t as template_effect, a6 as comment, a7 as text, w as get, a5 as user_derived, a0 as set_text, a8 as next } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { s as snippet } from './snippet-Bl5YVoCg.js';
import { B as Button } from './Button-CH2w4_A4.js';
import './clone-DaB-S2nH.js';
import './Image-CDw5qb5O.js';
import './misc-ByVveAau.js';
/* empty css                                             */
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
/* empty css                                                    */

var root_2 = from_html(`<img class="button-icon svelte-4ac0fl"/>`);
var root_1 = from_html(`<!> <!>`, 1);

function DownloadButton($$anchor, $$props) {
	push($$props, true);

	let elem_id = prop($$props, 'elem_id', 3, ""),
		elem_classes = prop($$props, 'elem_classes', 19, () => []),
		visible = prop($$props, 'visible', 3, true),
		variant = prop($$props, 'variant', 3, "secondary"),
		size = prop($$props, 'size', 3, "lg"),
		disabled = prop($$props, 'disabled', 3, false),
		scale = prop($$props, 'scale', 3, null),
		min_width = prop($$props, 'min_width', 3, undefined);

	function download_file() {
		$$props.on_click?.();

		if (!$$props.value?.url) {
			return;
		}

		let file_name;

		if (!$$props.value.orig_name && $$props.value.url) {
			const parts = $$props.value.url.split("/");

			file_name = parts[parts.length - 1];
			file_name = file_name.split("?")[0].split("#")[0];
		} else {
			file_name = $$props.value.orig_name;
		}

		const a = document.createElement("a");

		a.href = $$props.value.url;
		a.download = file_name || "file";
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
	}

	Button($$anchor, {
		get size() {
			return size();
		},

		get variant() {
			return variant();
		},

		get elem_id() {
			return elem_id();
		},

		get elem_classes() {
			return elem_classes();
		},

		get visible() {
			return visible();
		},
		onclick: download_file,
		get scale() {
			return scale();
		},

		get min_width() {
			return min_width();
		},

		get disabled() {
			return disabled();
		},

		children: ($$anchor, $$slotProps) => {
			var fragment_1 = root_1();
			var node = first_child(fragment_1);

			{
				var consequent = ($$anchor) => {
					var img = root_2();

					template_effect(() => {
						set_attribute(img, 'src', $$props.icon.url);
						set_attribute(img, 'alt', `${$$props.value} icon`);
					});

					append($$anchor, img);
				};

				if_block(node, ($$render) => {
					if ($$props.icon) $$render(consequent);
				});
			}

			var node_1 = sibling(node, 2);

			{
				var consequent_1 = ($$anchor) => {
					var fragment_2 = comment();
					var node_2 = first_child(fragment_2);

					snippet(node_2, () => $$props.children);
					append($$anchor, fragment_2);
				};

				if_block(node_1, ($$render) => {
					if ($$props.children) $$render(consequent_1);
				});
			}

			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);

	{
		let $0 = user_derived(() => !gradio.shared.interactive);

		DownloadButton($$anchor, {
			get value() {
				return gradio.props.value;
			},

			get variant() {
				return gradio.props.variant;
			},

			get elem_id() {
				return gradio.shared.elem_id;
			},

			get elem_classes() {
				return gradio.shared.elem_classes;
			},

			get size() {
				return gradio.props.size;
			},

			get scale() {
				return gradio.shared.scale;
			},

			get icon() {
				return gradio.props.icon;
			},

			get min_width() {
				return gradio.shared.min_width;
			},

			get visible() {
				return gradio.shared.visible;
			},

			get disabled() {
				return get($0);
			},
			on_click: () => gradio.dispatch("click"),
			children: ($$anchor, $$slotProps) => {
				next();

				var text$1 = text();

				template_effect(() => set_text(text$1, gradio.shared.label ?? ""));
				append($$anchor, text$1);
			},
			$$slots: { default: true }
		});
	}

	pop();
}

export { DownloadButton as BaseButton, Index as default };
//# sourceMappingURL=Index-CBNBXH2C.js.map
