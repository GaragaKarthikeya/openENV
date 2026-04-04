import { p as prop, i as if_block, a as set_attribute, b as set_class, f as set_style } from './i18n-CMo5lpzy.js';
import { R as push, a6 as comment, S as first_child, a as append, T as pop, aj as delegate, V as child, X as sibling, Y as reset, t as template_effect, W as from_html, w as get, a5 as user_derived } from './index-BvBk1Iap.js';
import { s as snippet } from './snippet-Bl5YVoCg.js';
import { I as Image } from './Image-CDw5qb5O.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
/* empty css                                                    */

var root_1 = from_html(`<a><!> <!></a>`);
var root_4 = from_html(`<button><!> <!></button>`);

function Button($$anchor, $$props) {
	push($$props, true);

	let elem_classes = prop($$props, 'elem_classes', 19, () => []),
		onclick = prop($$props, 'onclick', 3, () => {});

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent_2 = ($$anchor) => {
			var a = root_1();
			let classes;
			let styles;
			var node_1 = child(a);

			{
				var consequent = ($$anchor) => {
					{
						let $0 = user_derived(() => ({ alt: `${$$props.value} icon`, class: "button-icon" }));

						Image($$anchor, {
							get src() {
								return $$props.icon.url;
							},

							get restProps() {
								return get($0);
							}
						});
					}
				};

				if_block(node_1, ($$render) => {
					if ($$props.icon) $$render(consequent);
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

			reset(a);

			template_effect(
				($0) => {
					set_attribute(a, 'href', $$props.link);
					set_attribute(a, 'target', $$props.link_target);
					set_attribute(a, 'rel', $$props.link_target === "_blank" ? "noopener noreferrer" : undefined);
					set_attribute(a, 'aria-disabled', $$props.disabled);

					classes = set_class(a, 1, `${$$props.size ?? ''} ${$$props.variant ?? ''} ${$0 ?? ''}`, 'svelte-xzq5jh', classes, {
						hidden: $$props.visible === false || $$props.visible === "hidden",
						disabled: $$props.disabled
					});

					set_attribute(a, 'id', $$props.elem_id);

					styles = set_style(a, '', styles, {
						'flex-grow': $$props.scale,
						'pointer-events': $$props.disabled ? "none" : null,
						width: $$props.scale === 0 ? "fit-content" : null,
						'min-width': typeof $$props.min_width === "number" ? `calc(min(${$$props.min_width}px, 100%))` : null
					});
				},
				[() => elem_classes().join(' ')]
			);

			append($$anchor, a);
		};

		var alternate = ($$anchor) => {
			var button = root_4();

			button.__click = function (...$$args) {
				onclick()?.apply(this, $$args);
			};

			let classes_1;
			let styles_1;
			var node_4 = child(button);

			{
				var consequent_3 = ($$anchor) => {
					{
						let $0 = user_derived(() => ({ alt: `${$$props.value} icon` }));
						let $1 = user_derived(() => [`button-icon ${$$props.value ? "right-padded" : ""}`]);

						Image($$anchor, {
							get restProps() {
								return get($0);
							},

							get class_names() {
								return get($1);
							},

							get src() {
								return $$props.icon.url;
							}
						});
					}
				};

				if_block(node_4, ($$render) => {
					if ($$props.icon) $$render(consequent_3);
				});
			}

			var node_5 = sibling(node_4, 2);

			{
				var consequent_4 = ($$anchor) => {
					var fragment_4 = comment();
					var node_6 = first_child(fragment_4);

					snippet(node_6, () => $$props.children);
					append($$anchor, fragment_4);
				};

				if_block(node_5, ($$render) => {
					if ($$props.children) $$render(consequent_4);
				});
			}

			reset(button);

			template_effect(
				($0) => {
					classes_1 = set_class(button, 1, `${$$props.size ?? ''} ${$$props.variant ?? ''} ${$0 ?? ''}`, 'svelte-xzq5jh', classes_1, {
						hidden: $$props.visible === false || $$props.visible === "hidden"
					});

					set_attribute(button, 'id', $$props.elem_id);
					button.disabled = $$props.disabled;

					styles_1 = set_style(button, '', styles_1, {
						'flex-grow': $$props.scale,
						width: $$props.scale === 0 ? "fit-content" : null,
						'min-width': typeof $$props.min_width === "number" ? `calc(min(${$$props.min_width}px, 100%))` : null
					});
				},
				[() => elem_classes().join(' ')]
			);

			append($$anchor, button);
		};

		if_block(node, ($$render) => {
			if ($$props.link && $$props.link.length > 0) $$render(consequent_2); else $$render(alternate, false);
		});
	}

	append($$anchor, fragment);
	pop();
}

delegate(['click']);

export { Button as B };
//# sourceMappingURL=Button-CH2w4_A4.js.map
