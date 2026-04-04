import { p as prop, e as init, i as if_block, b as set_class, k as each, u as index } from './i18n-CMo5lpzy.js';
import { R as push, a6 as comment, S as first_child, a as append, T as pop, U as flushSync, V as child, Y as reset, t as template_effect, Z as event, x as set, W as from_html, a3 as mutable_source, a7 as text, a0 as set_text, w as get, z as untrack, X as sibling, G as deep_read_state } from './index-BvBk1Iap.js';

var root_4 = from_html(`<table class="svelte-wcwkqi"><tbody><tr class="svelte-wcwkqi"><td class="svelte-wcwkqi">Empty</td></tr></tbody></table>`);
var root_7 = from_html(`<td class="svelte-wcwkqi"> </td>`);
var root_8 = from_html(`<td class="svelte-wcwkqi">…</td>`);
var root_6 = from_html(`<tr class="svelte-wcwkqi"><!><!></tr>`);
var root_9 = from_html(`<div></div>`);
var root_5 = from_html(`<table class="svelte-wcwkqi"><tbody></tbody></table> <!>`, 1);
var root_1 = from_html(`<div><!></div>`);

function Example($$anchor, $$props) {
	push($$props, false);

	let value = prop($$props, 'value', 12);
	let type = prop($$props, 'type', 12);
	let selected = prop($$props, 'selected', 12, false);
	let index$1 = prop($$props, 'index', 12);
	let hovered = mutable_source(false);
	let loaded = Array.isArray(value());
	let is_empty = loaded && (value().length === 0 || value()[0].length === 0);

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

		get index() {
			return index$1();
		},

		set index($$value) {
			index$1($$value);
			flushSync();
		}
	};

	init();

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent_4 = ($$anchor) => {
			var div = root_1();
			let classes;
			var node_1 = child(div);

			{
				var consequent = ($$anchor) => {
					var text$1 = text();

					template_effect(() => set_text(text$1, value()));
					append($$anchor, text$1);
				};

				var alternate_1 = ($$anchor) => {
					var fragment_2 = comment();
					var node_2 = first_child(fragment_2);

					{
						var consequent_1 = ($$anchor) => {
							var table = root_4();

							append($$anchor, table);
						};

						var alternate = ($$anchor) => {
							var fragment_3 = root_5();
							var table_1 = first_child(fragment_3);
							var tbody = child(table_1);

							each(
								tbody,
								5,
								() => (
									deep_read_state(value()),
									untrack(() => value().slice(0, 3))
								),
								index,
								($$anchor, row) => {
									var tr = root_6();
									var node_3 = child(tr);

									each(node_3, 1, () => (get(row), untrack(() => get(row).slice(0, 3))), index, ($$anchor, cell) => {
										var td = root_7();
										var text_1 = child(td, true);

										reset(td);
										template_effect(() => set_text(text_1, get(cell)));
										append($$anchor, td);
									});

									var node_4 = sibling(node_3);

									{
										var consequent_2 = ($$anchor) => {
											var td_1 = root_8();

											append($$anchor, td_1);
										};

										if_block(node_4, ($$render) => {
											if ((get(row), untrack(() => get(row).length > 3))) $$render(consequent_2);
										});
									}

									reset(tr);
									append($$anchor, tr);
								}
							);

							reset(tbody);
							reset(table_1);

							var node_5 = sibling(table_1, 2);

							{
								var consequent_3 = ($$anchor) => {
									var div_1 = root_9();
									let classes_1;

									template_effect(() => classes_1 = set_class(div_1, 1, 'overlay svelte-wcwkqi', null, classes_1, {
										odd: index$1() % 2 != 0,
										even: index$1() % 2 == 0,
										button: type() === "gallery"
									}));

									append($$anchor, div_1);
								};

								if_block(node_5, ($$render) => {
									if ((
										deep_read_state(value()),
										untrack(() => value().length > 3)
									)) $$render(consequent_3);
								});
							}

							append($$anchor, fragment_3);
						};

						if_block(
							node_2,
							($$render) => {
								if (is_empty) $$render(consequent_1); else $$render(alternate, false);
							},
							true
						);
					}

					append($$anchor, fragment_2);
				};

				if_block(node_1, ($$render) => {
					if (typeof value() === "string") $$render(consequent); else $$render(alternate_1, false);
				});
			}

			reset(div);

			template_effect(() => classes = set_class(div, 1, 'svelte-wcwkqi', null, classes, {
				table: type() === "table",
				gallery: type() === "gallery",
				selected: selected()
			}));

			event('mouseenter', div, () => set(hovered, true));
			event('mouseleave', div, () => set(hovered, false));
			append($$anchor, div);
		};

		if_block(node, ($$render) => {
			if (loaded) $$render(consequent_4);
		});
	}

	append($$anchor, fragment);

	return pop($$exports);
}

export { Example as default };
//# sourceMappingURL=Example-DL_4juBu.js.map
