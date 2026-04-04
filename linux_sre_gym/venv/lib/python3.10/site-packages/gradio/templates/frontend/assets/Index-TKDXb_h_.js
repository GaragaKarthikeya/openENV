import { g as spread_props, d as bind_this, r as rest_props, b as set_class } from './i18n-CMo5lpzy.js';
import { R as push, y as user_effect, M as tick, S as first_child, w as get, t as template_effect, a as append, X as sibling, V as child, a5 as user_derived, T as pop, u as state, x as set, W as from_html, Y as reset } from './index-BvBk1Iap.js';
import { G as Gradio, s as should_show_scroll_fade } from './utils.svelte-CYLB_jIA.js';
import { M as Markdown } from './Index.svelte_svelte_type_style_lang-BiKot9ba.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import { B as Block } from './Block-Bx3BikCN.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { S as ScrollFade } from './ScrollFade-BJkuhu7T.js';
export { default as BaseExample } from './Example-YwgBtWi1.js';
import './clone-DaB-S2nH.js';
import './actions-Cit_UA5S.js';
import './Check-BlnBgjLT.js';
import './Copy-DFFray7x.js';
import './IconButtonWrapper-Ba0GidgU.js';
import './snippet-Bl5YVoCg.js';
import './MarkdownCode-Cg7CjFvQ.js';
import './html-CdUEtR5E.js';
import './prism-python-CbXMXbdR.js';
import './index-C7e-J7CF.js';
import './Clear-Dgs5UhTO.js';
/* empty css                                               */

var root_1 = from_html(`<!> <div><!></div> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);
	let wrapper;
	let show_fade = state(false);

	function update_fade() {
		if (!gradio.props.height) return;

		set(show_fade, should_show_scroll_fade(wrapper?.closest(".block")), true);
	}

	user_effect(() => {
		const container = wrapper?.closest(".block");

		if (!container || !gradio.props.height) return;

		container.addEventListener("scroll", update_fade);
		tick().then(update_fade);

		return () => container.removeEventListener("scroll", update_fade);
	});

	user_effect(() => {
		if (gradio.props.value !== undefined) tick().then(update_fade);
	});

	Block($$anchor, {
		get visible() {
			return gradio.shared.visible;
		},

		get elem_id() {
			return gradio.shared.elem_id;
		},

		get elem_classes() {
			return gradio.shared.elem_classes;
		},

		get container() {
			return gradio.shared.container;
		},
		allow_overflow: true,
		overflow_behavior: 'auto',
		get height() {
			return gradio.props.height;
		},

		get min_height() {
			return gradio.props.min_height;
		},

		get max_height() {
			return gradio.props.max_height;
		},

		get rtl() {
			return gradio.props.rtl;
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
					variant: 'center',
					on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
				}
			));

			var div = sibling(node, 2);
			let classes;
			var node_1 = child(div);

			{
				let $0 = user_derived(() => gradio.props.buttons?.includes("copy"));

				Markdown(node_1, {
					get value() {
						return gradio.props.value;
					},

					get elem_classes() {
						return gradio.shared.elem_classes;
					},

					get visible() {
						return gradio.shared.visible;
					},

					get rtl() {
						return gradio.props.rtl;
					},
					onchange: () => gradio.dispatch("change"),
					oncopy: (e) => gradio.dispatch("copy", e.detail),
					get latex_delimiters() {
						return gradio.props.latex_delimiters;
					},

					get sanitize_html() {
						return gradio.props.sanitize_html;
					},

					get line_breaks() {
						return gradio.props.line_breaks;
					},

					get header_links() {
						return gradio.props.header_links;
					},

					get show_copy_button() {
						return get($0);
					},

					get loading_status() {
						return gradio.shared.loading_status;
					},

					get theme_mode() {
						return gradio.shared.theme_mode;
					}
				});
			}

			reset(div);
			bind_this(div, ($$value) => wrapper = $$value, () => wrapper);

			var node_2 = sibling(div, 2);

			ScrollFade(node_2, {
				get visible() {
					return get(show_fade);
				}
			});

			template_effect(() => classes = set_class(div, 1, 'svelte-16ln60g', null, classes, {
				padding: gradio.props.padding,
				pending: gradio.shared.loading_status?.status === "pending" && gradio.shared.loading_status?.show_progress !== "hidden"
			}));

			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { Markdown as BaseMarkdown, Index as default };
//# sourceMappingURL=Index-TKDXb_h_.js.map
