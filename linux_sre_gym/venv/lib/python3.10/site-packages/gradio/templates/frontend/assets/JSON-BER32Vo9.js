import { p as prop, i as if_block, d as bind_this, b as set_class, f as set_style, a as set_attribute, k as each, u as index } from './i18n-CMo5lpzy.js';
import { a as append, f as from_svg, aj as delegate, R as push, u as state, v as proxy, y as user_effect, x as set, w as get, M as tick, V as child, X as sibling, Y as reset, t as template_effect, T as pop, S as first_child, a8 as next, a0 as set_text, a6 as comment, a5 as user_derived, W as from_html, at as to_array } from './index-BvBk1Iap.js';
import { C as Check } from './Check-BlnBgjLT.js';
import { C as Copy } from './Copy-DFFray7x.js';
import { I as IconButton } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { E as Empty } from './Empty-CHa-gxv7.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';

var root$1 = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--mdi" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24"><path fill="currentColor" d="M5 3h2v2H5v5a2 2 0 0 1-2 2a2 2 0 0 1 2 2v5h2v2H5c-1.07-.27-2-.9-2-2v-4a2 2 0 0 0-2-2H0v-2h1a2 2 0 0 0 2-2V5a2 2 0 0 1 2-2m14 0a2 2 0 0 1 2 2v4a2 2 0 0 0 2 2h1v2h-1a2 2 0 0 0-2 2v4a2 2 0 0 1-2 2h-2v-2h2v-5a2 2 0 0 1 2-2a2 2 0 0 1-2-2V5h-2V3h2m-7 12a1 1 0 0 1 1 1a1 1 0 0 1-1 1a1 1 0 0 1-1-1a1 1 0 0 1 1-1m-4 0a1 1 0 0 1 1 1a1 1 0 0 1-1 1a1 1 0 0 1-1-1a1 1 0 0 1 1-1m8 0a1 1 0 0 1 1 1a1 1 0 0 1-1 1a1 1 0 0 1-1-1a1 1 0 0 1 1-1Z"></path></svg>`);

function JSON$1($$anchor) {
	var svg = root$1();

	append($$anchor, svg);
}

var root_1$1 = from_html(`<button class="toggle svelte-1olemhd"></button>`);
var root_2 = from_html(`<span class="key svelte-1olemhd"> </span><span class="punctuation colon svelte-1olemhd">:</span>`, 1);
var root_4 = from_html(`<button class="preview svelte-1olemhd"> </button> <span> </span>`, 1);
var root_3 = from_html(`<span> </span> <!>`, 1);
var root_6 = from_html(`<span class="value string svelte-1olemhd"> </span>`);
var root_8 = from_html(`<span class="value number svelte-1olemhd"> </span>`);
var root_10 = from_html(`<span class="value bool svelte-1olemhd"> </span>`);
var root_12 = from_html(`<span class="value null svelte-1olemhd">null</span>`);
var root_13 = from_html(`<span> </span>`);
var root_14 = from_html(`<span class="punctuation svelte-1olemhd">,</span>`);
var root_17 = from_html(`<span class="punctuation svelte-1olemhd">,</span>`);
var root_15 = from_html(`<div><!> <div class="line svelte-1olemhd"><span class="line-number svelte-1olemhd"></span> <span class="content svelte-1olemhd"><span> </span> <!></span></div></div>`);
var root = from_html(`<div><div><span class="line-number svelte-1olemhd"></span> <span class="content svelte-1olemhd"><!> <!> <!> <!></span></div> <!></div>`);

function JSONNode($$anchor, $$props) {
	push($$props, true);

	let depth = prop($$props, 'depth', 3, 0),
		is_root = prop($$props, 'is_root', 3, false),
		is_last_item = prop($$props, 'is_last_item', 3, true),
		key = prop($$props, 'key', 3, null),
		open = prop($$props, 'open', 3, false),
		theme_mode = prop($$props, 'theme_mode', 3, "system"),
		show_indices = prop($$props, 'show_indices', 3, false),
		interactive = prop($$props, 'interactive', 3, true);

	let root_element = state(void 0);
	let collapsed = state(proxy(open() ? false : depth() >= 3));
	let child_nodes = state(proxy([]));

	function is_collapsible(val) {
		return val !== null && (typeof val === "object" || Array.isArray(val));
	}

	async function toggle_collapse() {
		set(collapsed, !get(collapsed));
		await tick();
	}

	function get_collapsed_preview(val) {
		if (Array.isArray(val)) return `Array(${val.length})`;
		if (typeof val === "object" && val !== null) return `Object(${Object.keys(val).length})`;

		return String(val);
	}

	user_effect(() => {
		if (is_collapsible($$props.value)) {
			set(child_nodes, Object.entries($$props.value), true);
		} else {
			set(child_nodes, [], true);
		}
	});

	function updateLineNumbers() {
		if (!get(root_element)) return;

		const lines = get(root_element).querySelectorAll(".line");

		lines.forEach((line, index) => {
			const line_number = line.querySelector(".line-number");

			if (line_number) {
				line_number.setAttribute("data-pseudo-content", (index + 1).toString());
				line_number?.setAttribute("aria-roledescription", `Line number ${index + 1}`);
				line_number?.setAttribute("title", `Line number ${index + 1}`);
			}
		});
	}

	user_effect(() => {
		$$props.value;
		get(collapsed);

		if (is_root() && get(root_element)) {
			tick().then(() => {
				updateLineNumbers();
			});
		}
	});

	var div = root();
	let classes;
	var div_1 = child(div);
	let classes_1;
	var span = sibling(child(div_1), 2);
	var node = child(span);

	{
		var consequent = ($$anchor) => {
			var button = root_1$1();

			button.__click = toggle_collapse;

			template_effect(() => {
				set_attribute(button, 'data-pseudo-content', interactive() ? get(collapsed) ? "▶" : "▼" : "");
				set_attribute(button, 'aria-label', get(collapsed) ? "Expand" : "Collapse");
				button.disabled = !interactive();
			});

			append($$anchor, button);
		};

		if_block(node, ($$render) => {
			if (is_collapsible($$props.value)) $$render(consequent);
		});
	}

	var node_1 = sibling(node, 2);

	{
		var consequent_1 = ($$anchor) => {
			var fragment = root_2();
			var span_1 = first_child(fragment);
			var text = child(span_1);

			reset(span_1);
			next();
			template_effect(() => set_text(text, `"${key() ?? ''}"`));
			append($$anchor, fragment);
		};

		if_block(node_1, ($$render) => {
			if (key() !== null) $$render(consequent_1);
		});
	}

	var node_2 = sibling(node_1, 2);

	{
		var consequent_3 = ($$anchor) => {
			var fragment_1 = root_3();
			var span_2 = first_child(fragment_1);
			let classes_2;
			var text_1 = child(span_2, true);

			reset(span_2);

			var node_3 = sibling(span_2, 2);

			{
				var consequent_2 = ($$anchor) => {
					var fragment_2 = root_4();
					var button_1 = first_child(fragment_2);

					button_1.__click = toggle_collapse;

					var text_2 = child(button_1, true);

					reset(button_1);

					var span_3 = sibling(button_1, 2);
					let classes_3;
					var text_3 = child(span_3, true);

					reset(span_3);

					template_effect(
						($0, $1, $2) => {
							set_text(text_2, $0);
							classes_3 = set_class(span_3, 1, 'punctuation bracket svelte-1olemhd', null, classes_3, $1);
							set_text(text_3, $2);
						},
						[
							() => get_collapsed_preview($$props.value),
							() => ({ 'square-bracket': Array.isArray($$props.value) }),
							() => Array.isArray($$props.value) ? "]" : "}"
						]
					);

					append($$anchor, fragment_2);
				};

				if_block(node_3, ($$render) => {
					if (get(collapsed)) $$render(consequent_2);
				});
			}

			template_effect(
				($0, $1) => {
					classes_2 = set_class(span_2, 1, 'punctuation bracket svelte-1olemhd', null, classes_2, $0);
					set_text(text_1, $1);
				},
				[
					() => ({ 'square-bracket': Array.isArray($$props.value) }),
					() => Array.isArray($$props.value) ? "[" : "{"
				]
			);

			append($$anchor, fragment_1);
		};

		var alternate_4 = ($$anchor) => {
			var fragment_3 = comment();
			var node_4 = first_child(fragment_3);

			{
				var consequent_4 = ($$anchor) => {
					var span_4 = root_6();
					var text_4 = child(span_4);

					reset(span_4);
					template_effect(() => set_text(text_4, `"${$$props.value ?? ''}"`));
					append($$anchor, span_4);
				};

				var alternate_3 = ($$anchor) => {
					var fragment_4 = comment();
					var node_5 = first_child(fragment_4);

					{
						var consequent_5 = ($$anchor) => {
							var span_5 = root_8();
							var text_5 = child(span_5, true);

							reset(span_5);
							template_effect(() => set_text(text_5, $$props.value));
							append($$anchor, span_5);
						};

						var alternate_2 = ($$anchor) => {
							var fragment_5 = comment();
							var node_6 = first_child(fragment_5);

							{
								var consequent_6 = ($$anchor) => {
									var span_6 = root_10();
									var text_6 = child(span_6, true);

									reset(span_6);
									template_effect(($0) => set_text(text_6, $0), [() => $$props.value.toString()]);
									append($$anchor, span_6);
								};

								var alternate_1 = ($$anchor) => {
									var fragment_6 = comment();
									var node_7 = first_child(fragment_6);

									{
										var consequent_7 = ($$anchor) => {
											var span_7 = root_12();

											append($$anchor, span_7);
										};

										var alternate = ($$anchor) => {
											var span_8 = root_13();
											var text_7 = child(span_8, true);

											reset(span_8);
											template_effect(() => set_text(text_7, $$props.value));
											append($$anchor, span_8);
										};

										if_block(
											node_7,
											($$render) => {
												if ($$props.value === null) $$render(consequent_7); else $$render(alternate, false);
											},
											true
										);
									}

									append($$anchor, fragment_6);
								};

								if_block(
									node_6,
									($$render) => {
										if (typeof $$props.value === "boolean") $$render(consequent_6); else $$render(alternate_1, false);
									},
									true
								);
							}

							append($$anchor, fragment_5);
						};

						if_block(
							node_5,
							($$render) => {
								if (typeof $$props.value === "number") $$render(consequent_5); else $$render(alternate_2, false);
							},
							true
						);
					}

					append($$anchor, fragment_4);
				};

				if_block(
					node_4,
					($$render) => {
						if (typeof $$props.value === "string") $$render(consequent_4); else $$render(alternate_3, false);
					},
					true
				);
			}

			append($$anchor, fragment_3);
		};

		if_block(node_2, ($$render) => {
			if (is_collapsible($$props.value)) $$render(consequent_3); else $$render(alternate_4, false);
		});
	}

	var node_8 = sibling(node_2, 2);

	{
		var consequent_8 = ($$anchor) => {
			var span_9 = root_14();

			append($$anchor, span_9);
		};

		if_block(node_8, ($$render) => {
			if (!is_last_item() && (!is_collapsible($$props.value) || get(collapsed))) $$render(consequent_8);
		});
	}

	reset(span);
	reset(div_1);

	var node_9 = sibling(div_1, 2);

	{
		var consequent_10 = ($$anchor) => {
			var div_2 = root_15();
			let classes_4;
			var node_10 = child(div_2);

			each(node_10, 17, () => get(child_nodes), index, ($$anchor, $$item, i) => {
				var $$array = user_derived(() => to_array(get($$item), 2));
				let subKey = () => get($$array)[0];
				let subVal = () => get($$array)[1];

				{
					let $0 = user_derived(() => depth() + 1);
					let $1 = user_derived(() => i === get(child_nodes).length - 1);
					let $2 = user_derived(() => Array.isArray($$props.value) && !show_indices() ? null : subKey());

					JSONNode($$anchor, {
						get value() {
							return subVal();
						},

						get depth() {
							return get($0);
						},

						get is_last_item() {
							return get($1);
						},

						get key() {
							return get($2);
						},

						get open() {
							return open();
						},

						get theme_mode() {
							return theme_mode();
						},

						get show_indices() {
							return show_indices();
						},

						get interactive() {
							return interactive();
						}
					});
				}
			});

			var div_3 = sibling(node_10, 2);
			var span_10 = sibling(child(div_3), 2);
			var span_11 = child(span_10);
			let classes_5;
			var text_8 = child(span_11, true);

			reset(span_11);

			var node_11 = sibling(span_11, 2);

			{
				var consequent_9 = ($$anchor) => {
					var span_12 = root_17();

					append($$anchor, span_12);
				};

				if_block(node_11, ($$render) => {
					if (!is_last_item()) $$render(consequent_9);
				});
			}

			reset(span_10);
			reset(div_3);
			reset(div_2);

			template_effect(
				($0, $1) => {
					classes_4 = set_class(div_2, 1, 'children svelte-1olemhd', null, classes_4, { hidden: get(collapsed) });
					classes_5 = set_class(span_11, 1, 'punctuation bracket svelte-1olemhd', null, classes_5, $0);
					set_text(text_8, $1);
				},
				[
					() => ({ 'square-bracket': Array.isArray($$props.value) }),
					() => Array.isArray($$props.value) ? "]" : "}"
				]
			);

			append($$anchor, div_2);
		};

		if_block(node_9, ($$render) => {
			if (is_collapsible($$props.value)) $$render(consequent_10);
		});
	}

	reset(div);
	bind_this(div, ($$value) => set(root_element, $$value), () => get(root_element));

	template_effect(() => {
		classes = set_class(div, 1, 'json-node svelte-1olemhd', null, classes, { root: is_root(), 'dark-mode': theme_mode() === "dark" });
		set_style(div, `--depth: ${depth() ?? ''};`);
		classes_1 = set_class(div_1, 1, 'line svelte-1olemhd', null, classes_1, { collapsed: get(collapsed) });
	});

	append($$anchor, div);
	pop();
}

delegate(['click']);

var root_1 = from_html(`<!> <div class="json-holder svelte-1lc38wd"><!></div>`, 1);
var root_5 = from_html(`<div class="empty-wrapper svelte-1lc38wd"><!></div>`);

function JSON_1($$anchor, $$props) {
	push($$props, true);

	let value = prop($$props, 'value', 19, () => ({})),
		open = prop($$props, 'open', 3, false),
		theme_mode = prop($$props, 'theme_mode', 3, "system"),
		show_indices = prop($$props, 'show_indices', 3, false),
		interactive = prop($$props, 'interactive', 3, true),
		show_copy_button = prop($$props, 'show_copy_button', 3, true),
		buttons = prop($$props, 'buttons', 3, null),
		on_custom_button_click = prop($$props, 'on_custom_button_click', 3, null);

	let json_max_height = user_derived(() => `calc(100% - ${$$props.label_height}px)`);
	let copied = state(false);
	let timer = state(void 0);

	function copy_feedback() {
		set(copied, true);

		if (get(timer)) clearTimeout(get(timer));

		set(
			timer,
			setTimeout(
				() => {
					set(copied, false);
				},
				1000
			),
			true
		);
	}

	async function handle_copy() {
		if ("clipboard" in navigator) {
			await navigator.clipboard.writeText(JSON.stringify(value(), null, 2));
			copy_feedback();
		}
	}

	function is_empty(obj) {
		return obj && Object.keys(obj).length === 0 && Object.getPrototypeOf(obj) === Object.prototype && JSON.stringify(obj) === JSON.stringify({});
	}

	user_effect(() => {
		return () => {
			if (get(timer)) clearTimeout(get(timer));
		};
	});

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent_2 = ($$anchor) => {
			var fragment_1 = root_1();
			var node_1 = first_child(fragment_1);

			{
				var consequent_1 = ($$anchor) => {
					IconButtonWrapper($$anchor, {
						get buttons() {
							return buttons();
						},

						get on_custom_button_click() {
							return on_custom_button_click();
						},

						children: ($$anchor, $$slotProps) => {
							var fragment_3 = comment();
							var node_2 = first_child(fragment_3);

							{
								var consequent = ($$anchor) => {
									{
										let $0 = user_derived(() => get(copied) ? "Copied" : "Copy");
										let $1 = user_derived(() => get(copied) ? Check : Copy);

										IconButton($$anchor, {
											show_label: false,
											get label() {
												return get($0);
											},

											get Icon() {
												return get($1);
											},
											onclick: () => handle_copy()
										});
									}
								};

								if_block(node_2, ($$render) => {
									if (show_copy_button()) $$render(consequent);
								});
							}

							append($$anchor, fragment_3);
						},
						$$slots: { default: true }
					});
				};

				if_block(node_1, ($$render) => {
					if (show_copy_button() || buttons() && buttons().some((btn) => typeof btn !== "string")) $$render(consequent_1);
				});
			}

			var div = sibling(node_1, 2);
			let styles;
			var node_3 = child(div);

			JSONNode(node_3, {
				get value() {
					return value();
				},
				depth: 0,
				is_root: true,
				get open() {
					return open();
				},

				get theme_mode() {
					return theme_mode();
				},

				get show_indices() {
					return show_indices();
				},

				get interactive() {
					return interactive();
				}
			});

			reset(div);
			template_effect(() => styles = set_style(div, '', styles, { 'max-height': get(json_max_height) }));
			append($$anchor, fragment_1);
		};

		var alternate = ($$anchor) => {
			var div_1 = root_5();
			var node_4 = child(div_1);

			Empty(node_4, {
				children: ($$anchor, $$slotProps) => {
					JSON$1($$anchor);
				},
				$$slots: { default: true }
			});

			reset(div_1);
			append($$anchor, div_1);
		};

		if_block(node, ($$render) => {
			if (value() && value() !== '""' && !is_empty(value())) $$render(consequent_2); else $$render(alternate, false);
		});
	}

	append($$anchor, fragment);
	pop();
}

export { JSON_1 as J, JSON$1 as a };
//# sourceMappingURL=JSON-BER32Vo9.js.map
