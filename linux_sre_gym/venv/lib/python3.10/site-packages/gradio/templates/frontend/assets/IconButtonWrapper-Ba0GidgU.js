import { p as prop, e as init, i as if_block, a as set_attribute, b as set_class, k as each, u as index } from './i18n-CMo5lpzy.js';
import { R as push, G as deep_read_state, z as untrack, t as template_effect, Z as event, a as append, T as pop, U as flushSync, V as child, a0 as set_text, W as from_html, Y as reset, X as sibling, a6 as comment, S as first_child, w as get } from './index-BvBk1Iap.js';
import { s as snippet } from './snippet-Bl5YVoCg.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';

var root_1 = from_html(`<span class="custom-button-label svelte-gnx6f7"> </span>`);
var root$1 = from_html(`<button class="custom-button svelte-gnx6f7"><!></button>`);

function CustomButton($$anchor, $$props) {
	push($$props, false);

	let button = prop($$props, 'button', 12);
	let on_click = prop($$props, 'on_click', 12);

	var $$exports = {
		get button() {
			return button();
		},

		set button($$value) {
			button($$value);
			flushSync();
		},

		get on_click() {
			return on_click();
		},

		set on_click($$value) {
			on_click($$value);
			flushSync();
		}
	};

	init();

	var button_1 = root$1();
	var node = child(button_1);

	{
		var consequent = ($$anchor) => {
			var span = root_1();
			var text = child(span, true);

			reset(span);
			template_effect(() => set_text(text, (deep_read_state(button()), untrack(() => button().value))));
			append($$anchor, span);
		};

		if_block(node, ($$render) => {
			if ((deep_read_state(button()), untrack(() => button().value))) $$render(consequent);
		});
	}

	reset(button_1);

	template_effect(() => {
		set_attribute(button_1, 'title', (
			deep_read_state(button()),
			untrack(() => button().value || "")
		));

		set_attribute(button_1, 'aria-label', (
			deep_read_state(button()),
			untrack(() => button().value || "Custom action")
		));
	});

	event('click', button_1, () => on_click()(button().id));
	append($$anchor, button_1);

	return pop($$exports);
}

var root = from_html(`<div><!> <!></div>`);

function IconButtonWrapper($$anchor, $$props) {
	push($$props, true);

	let top_panel = prop($$props, 'top_panel', 3, true),
		display_top_corner = prop($$props, 'display_top_corner', 3, false),
		show_background = prop($$props, 'show_background', 3, true),
		buttons = prop($$props, 'buttons', 3, null),
		on_custom_button_click = prop($$props, 'on_custom_button_click', 3, null);

	var div = root();
	var node = child(div);

	{
		var consequent = ($$anchor) => {
			var fragment = comment();
			var node_1 = first_child(fragment);

			snippet(node_1, () => $$props.children);
			append($$anchor, fragment);
		};

		if_block(node, ($$render) => {
			if ($$props.children) $$render(consequent);
		});
	}

	var node_2 = sibling(node, 2);

	{
		var consequent_2 = ($$anchor) => {
			var fragment_1 = comment();
			var node_3 = first_child(fragment_1);

			each(node_3, 17, buttons, index, ($$anchor, btn) => {
				var fragment_2 = comment();
				var node_4 = first_child(fragment_2);

				{
					var consequent_1 = ($$anchor) => {
						CustomButton($$anchor, {
							get button() {
								return get(btn);
							},

							on_click: (id) => {
								if (on_custom_button_click()) {
									on_custom_button_click()(id);
								}
							}
						});
					};

					if_block(node_4, ($$render) => {
						if (typeof get(btn) !== "string") $$render(consequent_1);
					});
				}

				append($$anchor, fragment_2);
			});

			append($$anchor, fragment_1);
		};

		if_block(node_2, ($$render) => {
			if (buttons()) $$render(consequent_2);
		});
	}

	reset(div);
	template_effect(() => set_class(div, 1, `icon-button-wrapper ${top_panel() ? "top-panel" : ""} ${display_top_corner() ? "display-top-corner" : "hide-top-corner"} ${!show_background() ? "no-background" : ""}`, 'svelte-1pnho82'));
	append($$anchor, div);
	pop();
}

export { CustomButton as C, IconButtonWrapper as I };
//# sourceMappingURL=IconButtonWrapper-Ba0GidgU.js.map
