import { i as if_block, s as slot, a as set_attribute, b as set_class, f as set_style, r as rest_props, g as spread_props } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, a as append, T as pop, X as sibling, W as from_html, V as child, w as get, Y as reset, a5 as user_derived } from './index-BvBk1Iap.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import './snippet-Bl5YVoCg.js';
import './Clear-Dgs5UhTO.js';
import './html-CdUEtR5E.js';
import './index-C7e-J7CF.js';
import './clone-DaB-S2nH.js';

var root = from_html(`<div><!> <!></div>`);

function Index($$anchor, $$props) {
	push($$props, true);

	// export let equal_height = true;
	// export let elem_id: string;
	// export let elem_classes: string[] = [];
	// export let visible: boolean | "hidden" = true;
	// export let variant: "default" | "panel" | "compact" = "default";
	// export let loading_status: LoadingStatus | undefined = undefined;
	// export let gradio: Gradio | undefined = undefined;
	// export let show_progress = false;
	// export let height: number | string | undefined;
	// export let min_height: number | string | undefined;
	// export let max_height: number | string | undefined;
	// export let scale: number | null = null;
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

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	let gradio = new Gradio(props);
	var div = root();
	let classes;
	let styles;
	var node = child(div);

	{
		var consequent = ($$anchor) => {
			{
				let $0 = user_derived(() => gradio.shared.loading_status
					? gradio.shared.loading_status.status == "pending" ? "generating" : gradio.shared.loading_status.status
					: null);

				Static($$anchor, spread_props(
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
						get status() {
							return get($0);
						}
					}
				));
			}
		};

		if_block(node, ($$render) => {
			if (gradio.shared.loading_status && gradio.shared.loading_status.show_progress && gradio) $$render(consequent);
		});
	}

	var node_1 = sibling(node, 2);

	slot(node_1, $$props, 'default', {});
	reset(div);

	template_effect(
		($0, $1) => {
			set_attribute(div, 'id', gradio.shared.elem_id);

			classes = set_class(div, 1, `row ${$0 ?? ''}`, 'svelte-7xavid', classes, {
				compact: gradio.props.variant === "compact",
				panel: gradio.props.variant === "panel",
				'unequal-height': gradio.props.equal_height === false,
				stretch: gradio.props.equal_height,
				hide: !gradio.shared.visible,
				'grow-children': gradio.shared.scale && gradio.shared.scale >= 1
			});

			styles = set_style(div, '', styles, $1);
		},
		[
			() => gradio.shared.elem_classes?.join(' '),
			() => ({
				height: get_dimension(gradio.props.height),
				'max-height': get_dimension(gradio.props.max_height),
				'min-height': get_dimension(gradio.props.min_height),
				'flex-grow': gradio.shared.scale
			})
		]
	);

	append($$anchor, div);
	pop();
}

export { Index as default };
//# sourceMappingURL=Index-DbuJYMQk.js.map
