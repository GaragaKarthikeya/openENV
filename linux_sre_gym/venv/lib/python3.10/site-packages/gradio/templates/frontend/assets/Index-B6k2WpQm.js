import { p as prop, s as slot, b as set_class, f as set_style, i as if_block, r as rest_props, g as spread_props } from './i18n-CMo5lpzy.js';
import { R as push, S as first_child, t as template_effect, a as append, T as pop, X as sibling, V as child, W as from_html, aj as delegate, a0 as set_text, Y as reset, a6 as comment, w as get, a5 as user_derived } from './index-BvBk1Iap.js';
import { B as Block } from './Block-Bx3BikCN.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import { B as BaseColumn } from './Index.svelte_svelte_type_style_lang-C97CbKPo.js';
import './snippet-Bl5YVoCg.js';
import './clone-DaB-S2nH.js';
import './Clear-Dgs5UhTO.js';
import './html-CdUEtR5E.js';
import './index-C7e-J7CF.js';

var root = from_html(`<button><span class="svelte-e5lyqv"> </span> <span class="icon svelte-e5lyqv">▼</span></button> <div data-testid="accordion-content"><!></div>`, 1);

function Accordion($$anchor, $$props) {
	push($$props, true);

	let open = prop($$props, 'open', 15, true),
		label = prop($$props, 'label', 3, "");

	var fragment = root();
	var button = first_child(fragment);

	button.__click = () => {
		open(!open());

		if (open()) {
			$$props.onexpand?.();
		} else {
			$$props.oncollapse?.();
		}
	};

	let classes;
	var span = child(button);
	var text = child(span, true);

	reset(span);

	var span_1 = sibling(span, 2);
	let styles;

	reset(button);

	var div = sibling(button, 2);
	let styles_1;
	var node = child(div);

	slot(node, $$props, 'default', {});
	reset(div);

	template_effect(() => {
		classes = set_class(button, 1, 'label-wrap svelte-e5lyqv', null, classes, { open: open() });
		set_text(text, label());
		styles = set_style(span_1, '', styles, { transform: open() ? "rotate(0)" : "rotate(90deg)" });
		styles_1 = set_style(div, '', styles_1, { display: open() ? "block" : "none" });
	});

	append($$anchor, fragment);
	pop();
}

delegate(['click']);

var root_1 = from_html(`<!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);

	class AccordionGradio extends Gradio {
		set_data(data) {
			if ("open" in data && data.open !== this.props.open) {
				if (data.open) {
					this.dispatch("expand");
					this.dispatch("gradio_expand");
				} else {
					this.dispatch("collapse");
				}
			}

			super.set_data(data);
			this.shared.loading_status.status = "complete";
		}
	}

	const gradio = new AccordionGradio(props);
	let label = user_derived(() => gradio.shared.label || "");
	let visibility = user_derived(() => gradio.shared.visible === true ? true : "hidden");

	Block($$anchor, {
		get elem_id() {
			return gradio.shared.elem_id;
		},

		get elem_classes() {
			return gradio.shared.elem_classes;
		},

		get visible() {
			return get(visibility);
		},

		children: ($$anchor, $$slotProps) => {
			var fragment_1 = root_1();
			var node = first_child(fragment_1);

			{
				var consequent = ($$anchor) => {
					Static($$anchor, spread_props(
						{
							get autoscroll() {
								return gradio.shared.autoscroll;
							},

							get i18n() {
								return gradio.i18n;
							}
						},
						() => gradio.shared.loading_status
					));
				};

				if_block(node, ($$render) => {
					if (gradio.shared.loading_status) $$render(consequent);
				});
			}

			var node_1 = sibling(node, 2);

			Accordion(node_1, {
				get label() {
					return get(label);
				},

				get open() {
					return gradio.props.open;
				},

				onexpand: () => {
					gradio.dispatch("expand");
					gradio.dispatch("gradio_expand");
				},
				oncollapse: () => gradio.dispatch("collapse"),
				children: ($$anchor, $$slotProps) => {
					BaseColumn($$anchor, {
						children: ($$anchor, $$slotProps) => {
							var fragment_4 = comment();
							var node_2 = first_child(fragment_4);

							slot(node_2, $$props, 'default', {});
							append($$anchor, fragment_4);
						},
						$$slots: { default: true }
					});
				},
				$$slots: { default: true }
			});

			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { Index as default };
//# sourceMappingURL=Index-B6k2WpQm.js.map
