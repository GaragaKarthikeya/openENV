import { p as prop, i as if_block, b as set_class, a as set_attribute, f as set_style } from './i18n-CMo5lpzy.js';
import { R as push, y as user_effect, t as template_effect, a as append, T as pop, X as sibling, W as from_html, V as child, w as get, a5 as user_derived, Y as reset, u as state, x as set } from './index-BvBk1Iap.js';
import { a as action } from './actions-Cit_UA5S.js';
import { c as css_units, a as copy } from './utils.svelte-CYLB_jIA.js';
import { C as Check } from './Check-BlnBgjLT.js';
import { C as Copy } from './Copy-DFFray7x.js';
import { I as IconButton } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { M as MarkdownCode } from './MarkdownCode-Cg7CjFvQ.js';

var root = from_html(`<div data-testid="markdown"><!> <!></div>`);

function Markdown($$anchor, $$props) {
	push($$props, true);

	let elem_classes = prop($$props, 'elem_classes', 19, () => []),
		visible = prop($$props, 'visible', 3, true),
		min_height = prop($$props, 'min_height', 3, undefined),
		rtl = prop($$props, 'rtl', 3, false),
		sanitize_html = prop($$props, 'sanitize_html', 3, true),
		line_breaks = prop($$props, 'line_breaks', 3, false),
		latex_delimiters = prop($$props, 'latex_delimiters', 19, () => []),
		header_links = prop($$props, 'header_links', 3, false),
		height = prop($$props, 'height', 3, undefined),
		show_copy_button = prop($$props, 'show_copy_button', 3, false),
		loading_status = prop($$props, 'loading_status', 3, undefined),
		onchange = prop($$props, 'onchange', 3, () => {}),
		oncopy = prop($$props, 'oncopy', 3, (val) => {});

	let copied = state(false);
	let timer;

	user_effect(() => {
		if ($$props.value) {
			onchange()();
		}
	});

	async function handle_copy() {
		if ("clipboard" in navigator) {
			await navigator.clipboard.writeText($$props.value);
			oncopy()({ value: $$props.value });
			copy_feedback();
		}
	}

	function copy_feedback() {
		set(copied, true);

		if (timer) clearTimeout(timer);

		timer = setTimeout(
			() => {
				set(copied, false);
			},
			1000
		);
	}

	var div = root();
	let classes;
	let styles;
	var node = child(div);

	{
		var consequent = ($$anchor) => {
			IconButtonWrapper($$anchor, {
				children: ($$anchor, $$slotProps) => {
					{
						let $0 = user_derived(() => get(copied) ? Check : Copy);
						let $1 = user_derived(() => get(copied) ? "Copied conversation" : "Copy conversation");

						IconButton($$anchor, {
							get Icon() {
								return get($0);
							},
							onclick: handle_copy,
							get label() {
								return get($1);
							}
						});
					}
				},
				$$slots: { default: true }
			});
		};

		if_block(node, ($$render) => {
			if (show_copy_button()) $$render(consequent);
		});
	}

	var node_1 = sibling(node, 2);

	MarkdownCode(node_1, {
		get message() {
			return $$props.value;
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
		chatbot: false,
		get header_links() {
			return header_links();
		},

		get theme_mode() {
			return $$props.theme_mode;
		}
	});

	reset(div);
	action(div, ($$node) => copy?.($$node));

	template_effect(
		($0, $1, $2) => {
			classes = set_class(div, 1, `prose ${$0 ?? ''}`, 'svelte-1xjkzpp', classes, { hide: !visible() });
			set_attribute(div, 'dir', rtl() ? "rtl" : "ltr");
			styles = set_style(div, $1, styles, $2);
			div.dir = div.dir;
		},
		[
			() => elem_classes()?.join(' ') || '',
			() => height()
				? `max-height: ${css_units(height())}; overflow-y: auto;`
				: "",

			() => ({
				'min-height': min_height() && loading_status()?.status !== "pending" ? css_units(min_height()) : undefined
			})
		]
	);

	append($$anchor, div);
	pop();
}

export { Markdown as M };
//# sourceMappingURL=Index.svelte_svelte_type_style_lang-BiKot9ba.js.map
