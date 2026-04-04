import { i as if_block, s as slot, d as bind_this, a as set_attribute, b as set_class, f as set_style, g as spread_props } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, w as get, a as append, T as pop, a5 as user_derived, X as sibling, W as from_html, V as child, Y as reset } from './index-BvBk1Iap.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';

var root = from_html(`<div><!> <!></div>`);

function BaseColumn($$anchor, $$props) {
	push($$props, true);
	let el;
	let scale = user_derived(() => $$props.scale ?? null);
	let min_width = user_derived(() => $$props.min_width ?? 0);
	let elem_id = user_derived(() => $$props.elem_id ?? "");
	let elem_classes = user_derived(() => $$props.elem_classes ?? []);
	let visible = user_derived(() => $$props.visible ?? true);
	let variant = user_derived(() => $$props.variant ?? "default");
	let loading_status = user_derived(() => $$props.loading_status);
	var div = root();
	let classes;
	let styles;
	var node = child(div);

	{
		var consequent = ($$anchor) => {
			{
				let $0 = user_derived(() => get(loading_status)
					? get(loading_status).status == "pending" ? "generating" : get(loading_status).status
					: null);

				Static($$anchor, spread_props(
					{
						get autoscroll() {
							return $$props.autoscroll;
						},

						get i18n() {
							return $$props.i18n;
						}
					},
					() => get(loading_status),
					{
						get status() {
							return get($0);
						}
					}
				));
			}
		};

		if_block(node, ($$render) => {
			if (get(loading_status) && get(loading_status).show_progress) $$render(consequent);
		});
	}

	var node_1 = sibling(node, 2);

	slot(node_1, $$props, 'default', {});
	reset(div);
	bind_this(div, ($$value) => el = $$value, () => el);

	template_effect(
		($0) => {
			set_attribute(div, 'id', get(elem_id));

			classes = set_class(div, 1, `column ${$0 ?? ''}`, 'svelte-siq5d6', classes, {
				compact: get(variant) === "compact",
				panel: get(variant) === "panel",
				hide: !get(visible)
			});

			styles = set_style(div, '', styles, {
				'flex-grow': get(scale),
				'min-width': `calc(min(${get(min_width) ?? ''}px, 100%))`
			});
		},
		[() => get(elem_classes).join(' ')]
	);

	append($$anchor, div);
	pop();
}

export { BaseColumn as B };
//# sourceMappingURL=Index.svelte_svelte_type_style_lang-C97CbKPo.js.map
