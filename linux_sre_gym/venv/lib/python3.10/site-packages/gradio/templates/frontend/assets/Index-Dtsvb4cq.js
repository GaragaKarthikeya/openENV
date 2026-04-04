import { p as prop, i as if_block, d as bind_this, f as set_style, a as set_attribute, k as each, b as set_class, r as rest_props } from './i18n-CMo5lpzy.js';
import { R as push, aa as onMount, t as template_effect, w as get, a0 as set_text, Z as event, a as append, T as pop, u as state, X as sibling, V as child, W as from_html, x as set, a5 as user_derived, Y as reset, S as first_child } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { h as html } from './html-CdUEtR5E.js';
import { P as Prism$1 } from './prism-python-CbXMXbdR.js';
import './clone-DaB-S2nH.js';

var prismTypescript = {};

var hasRequiredPrismTypescript;

function requirePrismTypescript () {
	if (hasRequiredPrismTypescript) return prismTypescript;
	hasRequiredPrismTypescript = 1;
	(function (Prism) {

		Prism.languages.typescript = Prism.languages.extend('javascript', {
			'class-name': {
				pattern: /(\b(?:class|extends|implements|instanceof|interface|new|type)\s+)(?!keyof\b)(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?:\s*<(?:[^<>]|<(?:[^<>]|<[^<>]*>)*>)*>)?/,
				lookbehind: true,
				greedy: true,
				inside: null // see below
			},
			'builtin': /\b(?:Array|Function|Promise|any|boolean|console|never|number|string|symbol|unknown)\b/,
		});

		// The keywords TypeScript adds to JavaScript
		Prism.languages.typescript.keyword.push(
			/\b(?:abstract|declare|is|keyof|readonly|require)\b/,
			// keywords that have to be followed by an identifier
			/\b(?:asserts|infer|interface|module|namespace|type)\b(?=\s*(?:[{_$a-zA-Z\xA0-\uFFFF]|$))/,
			// This is for `import type *, {}`
			/\btype\b(?=\s*(?:[\{*]|$))/
		);

		// doesn't work with TS because TS is too complex
		delete Prism.languages.typescript['parameter'];
		delete Prism.languages.typescript['literal-property'];

		// a version of typescript specifically for highlighting types
		var typeInside = Prism.languages.extend('typescript', {});
		delete typeInside['class-name'];

		Prism.languages.typescript['class-name'].inside = typeInside;

		Prism.languages.insertBefore('typescript', 'function', {
			'decorator': {
				pattern: /@[$\w\xA0-\uFFFF]+/,
				inside: {
					'at': {
						pattern: /^@/,
						alias: 'operator'
					},
					'function': /^[\s\S]+/
				}
			},
			'generic-function': {
				// e.g. foo<T extends "bar" | "baz">( ...
				pattern: /#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*\s*<(?:[^<>]|<(?:[^<>]|<[^<>]*>)*>)*>(?=\s*\()/,
				greedy: true,
				inside: {
					'function': /^#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*/,
					'generic': {
						pattern: /<[\s\S]+/, // everything after the first <
						alias: 'class-name',
						inside: typeInside
					}
				}
			}
		});

		Prism.languages.ts = Prism.languages.typescript;

	}(Prism));
	return prismTypescript;
}

requirePrismTypescript();

var root_1 = from_html(`<span class="title svelte-1kuiw39"> </span>`);
var root_4 = from_html(`<a class="param-link svelte-1kuiw39"><span class="link-icon svelte-1kuiw39">🔗</span></a>`);
var root_5 = from_html(`: <!>`, 1);
var root_6 = from_html(`<div><span class="svelte-1kuiw39">default</span> <code class="svelte-1kuiw39">= <!></code></div>`);
var root_7 = from_html(`<div class="description svelte-1kuiw39"><p><!></p></div>`);
var root_3 = from_html(`<details class="param md svelte-1kuiw39"><summary class="type svelte-1kuiw39"><!> <pre><code class="svelte-1kuiw39"> <!></code></pre></summary> <!> <!></details>`);
var root_2 = from_html(`<div class="param-content svelte-1kuiw39"></div>`);
var root = from_html(`<div class="wrap svelte-1kuiw39"><div class="header svelte-1kuiw39"><!> <button class="toggle-all svelte-1kuiw39"> </button></div> <!></div>`);

function ParamViewer($$anchor, $$props) {
	push($$props, true);

	let linkify = prop($$props, 'linkify', 19, () => []);
	let component_root;
	let all_open = state(false);
	let lang = "python";
	let _docs = user_derived(() => highlight_code($$props.docs, lang));

	function create_slug(name, anchor_links) {
		let prefix = "param-";

		if (typeof anchor_links === "string") {
			prefix += anchor_links + "-";
		}

		return prefix + name.toLowerCase().replace(/[^a-z0-9]+/g, "-");
	}

	function highlight(code, lang) {
		let highlighted = Prism$1.highlight(code, Prism$1.languages[lang], lang);

		for (const link of linkify()) {
			highlighted = highlighted.replace(new RegExp(link, "g"), `<a href="#h-${link.toLocaleLowerCase()}">${link}</a>`);
		}

		return highlighted;
	}

	function highlight_code(_docs, lang) {
		if (!_docs) {
			return [];
		}

		return Object.entries(_docs).map(([name, { type, description, default: _default }]) => {
			let highlighted_type = type ? highlight(type, lang) : null;

			return {
				name,
				type: highlighted_type,
				description,
				default: _default ? highlight(_default, lang) : null
			};
		});
	}

	function toggle_all() {
		set(all_open, !get(all_open));

		const details = component_root.querySelectorAll(".param");

		details.forEach((detail) => {
			if (detail instanceof HTMLDetailsElement) {
				detail.open = get(all_open);
			}
		});
	}

	function render_links(description) {
		const escaped = description.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
		const markdown_links = escaped.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

		return markdown_links;
	}

	onMount(() => {
		if (window.location.hash) {
			open_parameter_from_hash(window.location.hash);
		}

		window.addEventListener("hashchange", (e) => {
			open_parameter_from_hash(window.location.hash);
		});
	});

	function open_parameter_from_hash(hash) {
		if (!component_root) return;

		const id = hash.slice(1);
		const detail = component_root.querySelector(`#${id}`);

		if (detail instanceof HTMLDetailsElement) {
			detail.open = true;
			detail.scrollIntoView({ behavior: "smooth" });
		}
	}

	const get_dimension = (dimension_value) => {
		if (dimension_value === undefined) {
			return undefined;
		}

		if (typeof dimension_value === "number") {
			return dimension_value + "px";
		} else if (typeof dimension_value === "string") {
			return dimension_value;
		}
	};

	var div = root();
	let styles;
	var div_1 = child(div);
	var node = child(div_1);

	{
		var consequent = ($$anchor) => {
			var span = root_1();
			var text = child(span, true);

			reset(span);
			template_effect(() => set_text(text, $$props.header));
			append($$anchor, span);
		};

		if_block(node, ($$render) => {
			if ($$props.header !== null) $$render(consequent);
		});
	}

	var button = sibling(node, 2);
	var text_1 = child(button, true);

	reset(button);
	reset(div_1);

	var node_1 = sibling(div_1, 2);

	{
		var consequent_5 = ($$anchor) => {
			var div_2 = root_2();

			each(div_2, 21, () => get(_docs), ({ type, description, default: _default, name }) => name, ($$anchor, $$item) => {
				let type = () => get($$item).type;
				let description = () => get($$item).description;
				let _default = () => get($$item).default;
				let name = () => get($$item).name;
				var details_1 = root_3();
				var summary = child(details_1);
				var node_2 = child(summary);

				{
					var consequent_1 = ($$anchor) => {
						var a = root_4();

						template_effect(($0) => set_attribute(a, 'href', `#${$0 ?? ''}`), [() => create_slug(name() || '', $$props.anchor_links)]);
						append($$anchor, a);
					};

					if_block(node_2, ($$render) => {
						if ($$props.anchor_links) $$render(consequent_1);
					});
				}

				var pre = sibling(node_2, 2);

				set_class(pre, 1, 'language-python svelte-1kuiw39');

				var code_1 = child(pre);
				var text_2 = child(code_1, true);
				var node_3 = sibling(text_2);

				{
					var consequent_2 = ($$anchor) => {
						var fragment = root_5();
						var node_4 = sibling(first_child(fragment));

						html(node_4, type);
						append($$anchor, fragment);
					};

					if_block(node_3, ($$render) => {
						if (type()) $$render(consequent_2);
					});
				}

				reset(code_1);
				reset(pre);
				reset(summary);

				var node_5 = sibling(summary, 2);

				{
					var consequent_3 = ($$anchor) => {
						var div_3 = root_6();
						let classes;
						var span_1 = child(div_3);

						set_style(span_1, '', {}, { 'padding-right': "4px" });

						var code_2 = sibling(span_1, 2);
						var node_6 = sibling(child(code_2));

						html(node_6, _default);
						reset(code_2);
						reset(div_3);
						template_effect(() => classes = set_class(div_3, 1, 'default svelte-1kuiw39', null, classes, { last: !description() }));
						append($$anchor, div_3);
					};

					if_block(node_5, ($$render) => {
						if (_default()) $$render(consequent_3);
					});
				}

				var node_7 = sibling(node_5, 2);

				{
					var consequent_4 = ($$anchor) => {
						var div_4 = root_7();
						var p = child(div_4);
						var node_8 = child(p);

						html(node_8, () => render_links(description()));
						reset(p);
						reset(div_4);
						append($$anchor, div_4);
					};

					if_block(node_7, ($$render) => {
						if (description()) $$render(consequent_4);
					});
				}

				reset(details_1);

				template_effect(
					($0) => {
						set_attribute(details_1, 'id', $0);
						set_text(text_2, name());
					},
					[
						() => $$props.anchor_links
							? create_slug(name() || "", $$props.anchor_links)
							: undefined
					]
				);

				append($$anchor, details_1);
			});

			reset(div_2);
			append($$anchor, div_2);
		};

		if_block(node_1, ($$render) => {
			if (get(_docs)) $$render(consequent_5);
		});
	}

	reset(div);
	bind_this(div, ($$value) => component_root = $$value, () => component_root);

	template_effect(
		($0) => {
			styles = set_style(div, '', styles, $0);
			set_attribute(button, 'title', get(all_open) ? "Close All" : "Open All");
			set_text(text_1, get(all_open) ? "▲" : "▼");
		},
		[() => ({ 'max-height': get_dimension($$props.max_height) })]
	);

	event('click', button, toggle_all);
	append($$anchor, div);
	pop();
}

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);

	ParamViewer($$anchor, {
		get docs() {
			return gradio.props.value;
		},

		get linkify() {
			return gradio.props.linkify;
		},

		get header() {
			return gradio.props.header;
		},

		get anchor_links() {
			return gradio.props.anchor_links;
		},

		get max_height() {
			return gradio.props.max_height;
		}
	});

	pop();
}

export { Index as default };
//# sourceMappingURL=Index-Dtsvb4cq.js.map
