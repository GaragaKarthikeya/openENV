import { p as prop, i as if_block, b as set_class, t as remove_input_defaults } from './i18n-CMo5lpzy.js';
import { R as push, u as state, v as proxy, y as user_effect, w as get, x as set, t as template_effect, Z as event, a as append, T as pop, V as child, X as sibling, W as from_html, a5 as user_derived, Y as reset, a0 as set_text } from './index-BvBk1Iap.js';
import { s as snapshot } from './clone-DaB-S2nH.js';
import { a as bind_checked } from './input-C317uaXR.js';

var root_1 = from_html(`<span class="label-text svelte-1q8xtp9"> </span>`);
var root = from_html(`<label><input type="checkbox" name="test" data-testid="checkbox" class="svelte-1q8xtp9"/> <!></label>`);

function Checkbox($$anchor, $$props) {
	push($$props, true);

	let label = prop($$props, 'label', 3, "Checkbox"),
		value = prop($$props, 'value', 15),
		interactive = prop($$props, 'interactive', 3, true),
		show_label = prop($$props, 'show_label', 3, true);

	let disabled = user_derived(() => !interactive());
	let old_value = state(proxy(value()));

	user_effect(() => {
		if (get(old_value) !== value()) {
			set(old_value, value(), true);
			$$props.on_change?.(snapshot(value()));
		}
	});

	async function handle_enter(event) {
		if (event.key === "Enter") {
			value(!value());

			$$props.on_select?.({
				index: 0,
				value: event.currentTarget.checked,
				selected: event.currentTarget.checked
			});
		}
	}

	async function handle_input(event) {
		value(event.currentTarget.checked);

		$$props.on_select?.({
			index: 0,
			value: event.currentTarget.checked,
			selected: event.currentTarget.checked
		});

		$$props.on_input?.();
	}

	var label_1 = root();
	let classes;
	var input = child(label_1);

	remove_input_defaults(input);

	var node = sibling(input, 2);

	{
		var consequent = ($$anchor) => {
			var span = root_1();
			var text = child(span, true);

			reset(span);
			template_effect(() => set_text(text, label()));
			append($$anchor, span);
		};

		if_block(node, ($$render) => {
			if (show_label()) $$render(consequent);
		});
	}

	reset(label_1);

	template_effect(() => {
		classes = set_class(label_1, 1, 'checkbox-container svelte-1q8xtp9', null, classes, { disabled: get(disabled) });
		input.disabled = get(disabled);
	});

	bind_checked(input, value);
	event('keydown', input, handle_enter);
	event('input', input, handle_input);
	append($$anchor, label_1);
	pop();
}

export { Checkbox as C };
//# sourceMappingURL=Checkbox-BwK0lQJ8.js.map
