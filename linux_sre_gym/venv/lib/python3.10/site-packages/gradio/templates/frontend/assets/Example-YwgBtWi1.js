import { p as prop, b as set_class } from './i18n-CMo5lpzy.js';
import { R as push, w as get, t as template_effect, a as append, T as pop, U as flushSync, V as child, ad as derived_safe_equal, G as deep_read_state, z as untrack, W as from_html, Y as reset } from './index-BvBk1Iap.js';
import { M as MarkdownCode } from './MarkdownCode-Cg7CjFvQ.js';
/* empty css                                               */
import './html-CdUEtR5E.js';
import './prism-python-CbXMXbdR.js';
import './index-C7e-J7CF.js';

var root = from_html(`<div><!></div>`);

function Example($$anchor, $$props) {
	push($$props, false);

	let value = prop($$props, 'value', 12);
	let type = prop($$props, 'type', 12);
	let selected = prop($$props, 'selected', 12, false);
	let sanitize_html = prop($$props, 'sanitize_html', 12);
	let line_breaks = prop($$props, 'line_breaks', 12);
	let latex_delimiters = prop($$props, 'latex_delimiters', 12);

	function truncate_text(text, max_length = 60) {
		if (!text) return "";

		const str = String(text);

		if (str.length <= max_length) return str;

		return str.slice(0, max_length) + "...";
	}

	var $$exports = {
		get value() {
			return value();
		},

		set value($$value) {
			value($$value);
			flushSync();
		},

		get type() {
			return type();
		},

		set type($$value) {
			type($$value);
			flushSync();
		},

		get selected() {
			return selected();
		},

		set selected($$value) {
			selected($$value);
			flushSync();
		},

		get sanitize_html() {
			return sanitize_html();
		},

		set sanitize_html($$value) {
			sanitize_html($$value);
			flushSync();
		},

		get line_breaks() {
			return line_breaks();
		},

		set line_breaks($$value) {
			line_breaks($$value);
			flushSync();
		},

		get latex_delimiters() {
			return latex_delimiters();
		},

		set latex_delimiters($$value) {
			latex_delimiters($$value);
			flushSync();
		}
	};

	var div = root();
	let classes;
	var node = child(div);

	{
		let $0 = derived_safe_equal(() => (
			deep_read_state(value()),
			untrack(() => truncate_text(value()))
		));

		MarkdownCode(node, {
			get message() {
				return get($0);
			},

			get latex_delimiters() {
				return latex_delimiters();
			},

			get sanitize_html() {
				return sanitize_html();
			},

			get line_breaks() {
				return line_breaks();
			},
			chatbot: false
		});
	}

	reset(div);

	template_effect(() => classes = set_class(div, 1, 'prose svelte-11ua876', null, classes, {
		table: type() === "table",
		gallery: type() === "gallery",
		selected: selected()
	}));

	append($$anchor, div);

	return pop($$exports);
}

export { Example as default };
//# sourceMappingURL=Example-YwgBtWi1.js.map
