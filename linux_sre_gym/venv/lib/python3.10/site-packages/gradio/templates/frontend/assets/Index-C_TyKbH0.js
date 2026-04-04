import { p as prop, d as bind_this, i as if_block, a as set_attribute, r as rest_props } from './i18n-CMo5lpzy.js';
import { R as push, S as first_child, a as append, t as template_effect, T as pop, X as sibling, W as from_html, au as prepare_files, M as tick, aj as delegate, a6 as comment, w as get, a5 as user_derived, v as proxy, y as user_effect, a7 as text, a0 as set_text, a8 as next } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { s as snippet } from './snippet-Bl5YVoCg.js';
import { B as Button } from './Button-CH2w4_A4.js';
import './clone-DaB-S2nH.js';
import './Image-CDw5qb5O.js';
import './misc-ByVveAau.js';
/* empty css                                             */
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
/* empty css                                                    */

var root_3 = from_html(`<img class="button-icon svelte-94gmgt"/>`);
var root_2 = from_html(`<!> <!>`, 1);
var root_1 = from_html(`<input class="hide svelte-94gmgt" type="file"/> <!>`, 1);

function UploadButton($$anchor, $$props) {
	push($$props, true);

	let elem_id = prop($$props, 'elem_id', 3, ""),
		elem_classes = prop($$props, 'elem_classes', 19, () => []),
		visible = prop($$props, 'visible', 3, true),
		file_types = prop($$props, 'file_types', 19, () => []),
		size = prop($$props, 'size', 3, "lg"),
		icon = prop($$props, 'icon', 3, null),
		scale = prop($$props, 'scale', 3, null),
		min_width = prop($$props, 'min_width', 3, undefined),
		variant = prop($$props, 'variant', 3, "secondary"),
		disabled = prop($$props, 'disabled', 3, false),
		max_file_size = prop($$props, 'max_file_size', 3, null);

	let hidden_upload;

	let accept_file_types = user_derived(() => {
		if (file_types() == null) {
			return null;
		}

		const mapped = file_types().map((x) => {
			if (x.startsWith(".")) {
				return x;
			}

			return x + "/*";
		});

		return mapped.join(", ");
	});

	function open_file_upload() {
		$$props.onclick?.();
		hidden_upload.click();
	}

	async function load_files(files) {
		let _files = Array.from(files);

		if (!files.length) {
			return;
		}

		if ($$props.file_count === "single") {
			_files = [files[0]];
		}

		let all_file_data = await prepare_files(_files);

		await tick();

		try {
			all_file_data = (await $$props.upload(all_file_data, $$props.root, undefined, max_file_size() ?? Infinity))?.filter((x) => x !== null);
		} catch(e) {
			$$props.onerror?.(e.message);

			return;
		}

		const new_value = $$props.file_count === "single" ? all_file_data?.[0] : all_file_data;

		$$props.onchange?.(new_value);
		$$props.onupload?.(new_value);
	}

	async function load_files_from_upload(e) {
		const target = e.target;

		if (!target.files) return;

		await load_files(target.files);
	}

	function clear_input_value(e) {
		const target = e.target;

		if (target.value) target.value = "";
	}

	var fragment = root_1();
	var input = first_child(fragment);

	input.__change = load_files_from_upload;
	input.__click = clear_input_value;
	bind_this(input, ($$value) => hidden_upload = $$value, () => hidden_upload);

	var node = sibling(input, 2);

	Button(node, {
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
		onclick: open_file_upload,
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
			var fragment_1 = root_2();
			var node_1 = first_child(fragment_1);

			{
				var consequent = ($$anchor) => {
					var img = root_3();

					template_effect(() => {
						set_attribute(img, 'src', icon().url);
						set_attribute(img, 'alt', `${$$props.value} icon`);
					});

					append($$anchor, img);
				};

				if_block(node_1, ($$render) => {
					if (icon()) $$render(consequent);
				});
			}

			var node_2 = sibling(node_1, 2);

			{
				var consequent_1 = ($$anchor) => {
					var fragment_2 = comment();
					var node_3 = first_child(fragment_2);

					snippet(node_3, () => $$props.children);
					append($$anchor, fragment_2);
				};

				if_block(node_2, ($$render) => {
					if ($$props.children) $$render(consequent_1);
				});
			}

			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	template_effect(() => {
		set_attribute(input, 'accept', get(accept_file_types));
		input.multiple = $$props.file_count === "multiple" || undefined;
		input.webkitdirectory = $$props.file_count === "directory" || undefined;
		set_attribute(input, 'mozdirectory', $$props.file_count === "directory" || undefined);
		set_attribute(input, 'data-testid', `${$$props.label ?? ''}-upload-button`);
	});

	append($$anchor, fragment);
	pop();
}

delegate(['change', 'click']);

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);
	let value = proxy(gradio.props.value);

	user_effect(() => {
		if (value !== gradio.props.value) {
			gradio.props.value = value;
		}
	});

	async function handle_event(detail, event) {
		gradio.props.value = detail;
		gradio.dispatch(event);
	}

	const disabled = user_derived(() => !gradio.shared.interactive);

	UploadButton($$anchor, {
		get elem_id() {
			return gradio.shared.elem_id;
		},

		get elem_classes() {
			return gradio.shared.elem_classes;
		},

		get visible() {
			return gradio.shared.visible;
		},

		get file_count() {
			return gradio.props.file_count;
		},

		get file_types() {
			return gradio.props.file_types;
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

		get root() {
			return gradio.shared.root;
		},

		get value() {
			return value;
		},

		get disabled() {
			return get(disabled);
		},

		get variant() {
			return gradio.props.variant;
		},

		get label() {
			return gradio.shared.label;
		},

		get max_file_size() {
			return gradio.shared.max_file_size;
		},
		onclick: () => gradio.dispatch("click"),
		onchange: (detail) => handle_event(detail, "change"),
		onupload: (detail) => handle_event(detail, "upload"),
		onerror: (detail) => {
			gradio.dispatch("error", detail);
		},
		upload: (...args) => gradio.shared.client.upload(...args),
		children: ($$anchor, $$slotProps) => {
			next();

			var text$1 = text();

			template_effect(() => set_text(text$1, gradio.shared.label ?? ""));
			append($$anchor, text$1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { UploadButton as BaseUploadButton, Index as default };
//# sourceMappingURL=Index-C_TyKbH0.js.map
