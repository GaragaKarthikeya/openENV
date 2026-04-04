import { p as prop, e as init, a as set_attribute, i as if_block, b as set_class, k as each, d as bind_this, f as set_style, u as index, r as rest_props, g as spread_props } from './i18n-CMo5lpzy.js';
import { R as push, a4 as createEventDispatcher, V as child, X as sibling, Y as reset, t as template_effect, a0 as set_text, Z as event, a as append, T as pop, U as flushSync, W as from_html, u as state, v as proxy, y as user_effect, x as set, w as get, aa as onMount, a7 as text, aw as $window, a5 as user_derived, a6 as comment, S as first_child, a8 as next, ak as remove_textarea_child, M as tick } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { t as transition, f as fade } from './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import { b as bind_value } from './input-C317uaXR.js';
import { I as IconButton } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { B as BlockTitle } from './BlockTitle-7XQ-8ENg.js';
import { C as Check } from './Check-BlnBgjLT.js';
import { C as Copy } from './Copy-DFFray7x.js';
import { P as Plus } from './Plus-CFoCxzu1.js';
import { S as Send } from './Send-Ee555fP_.js';
import { T as Trash } from './Trash-CGTLExrZ.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { b as Dropdown, D as DropdownOptions } from './Dropdown-DUCTuRQm.js';
/* empty css                                               */
import { B as Block } from './Block-Bx3BikCN.js';
import { S as Static } from './index-Lz7fsHFt.js';
export { default as BaseExample } from './Example-BIlLUrfo.js';
import './clone-DaB-S2nH.js';
import './html-CdUEtR5E.js';
import './index-C7e-J7CF.js';
import './snippet-Bl5YVoCg.js';
import './Info-BrkJhX6_.js';
import './window-BGXjFJtG.js';
import './DropdownArrow-Dw1NZKCM.js';
import './Clear-Dgs5UhTO.js';

var root$1 = from_html(`<div class="s s--slider svelte-1io85vl" style="font-size:var(--font-size-sm)px"><button role="switch" class="svelte-1io85vl"></button> <span> </span></div>`);

function Switch($$anchor, $$props) {
	push($$props, false);

	// based on https://svelte.dev/playground/d65a4e9f0ae74d1eb1b08d13e428af32?version=3.35.0
	let label = prop($$props, 'label', 12);

	let checked = prop($$props, 'checked', 12, false);
	let disabled = prop($$props, 'disabled', 12, false);
	const dispatch = createEventDispatcher();

	// @ts-ignore
	function handleClick(event) {
		const target = event.target;

		// @ts-ignore
		const state = target.getAttribute("aria-checked");

		checked(state === "true" ? false : true);
		dispatch("click", { checked: checked() });
	}

	var $$exports = {
		get label() {
			return label();
		},

		set label($$value) {
			label($$value);
			flushSync();
		},

		get checked() {
			return checked();
		},

		set checked($$value) {
			checked($$value);
			flushSync();
		},

		get disabled() {
			return disabled();
		},

		set disabled($$value) {
			disabled($$value);
			flushSync();
		}
	};

	init();

	var div = root$1();
	var button = child(div);
	var span = sibling(button, 2);
	var text = child(span, true);

	reset(span);
	reset(div);

	template_effect(() => {
		set_attribute(button, 'aria-checked', checked());
		button.disabled = disabled();
		set_text(text, label());
	});

	event('click', button, handleClick);
	append($$anchor, div);

	return pop($$exports);
}

var root_5 = from_html(`<div><!></div>`);
var root_7 = from_html(`<div class="loading-overlay svelte-1p54cvv"><div class="loading-spinner svelte-1p54cvv"></div> <div class="loading-text svelte-1p54cvv">Converting to dialogue format...</div></div>`);
var root_9 = from_html(`<textarea rows="1" readonly="" class="svelte-1p54cvv"></textarea>`);
var root_11 = from_html(`<div id="tag-menu" class="tag-menu svelte-1p54cvv"><!></div>`);
var root_12 = from_html(`<div><button class="add-button svelte-1p54cvv" aria-label="Add new line"><!></button></div>`);
var root_8 = from_html(`<div class="dialogue-line svelte-1p54cvv"><div class="speaker-column svelte-1p54cvv" role="button" tabindex="0"><!></div> <div class="text-column svelte-1p54cvv"><div class="input-container svelte-1p54cvv"><textarea rows="1" class="svelte-1p54cvv"></textarea> <!></div></div> <!> <div><button class="delete-button svelte-1p54cvv" aria-label="Remove current line"><!></button></div></div>`);
var root_6 = from_html(`<div><!> <!></div>`);
var root_15 = from_html(`<div class="loading-overlay svelte-1p54cvv"><div class="loading-spinner svelte-1p54cvv"></div> <div class="loading-text svelte-1p54cvv">Converting to plain text...</div></div>`);
var root_16 = from_html(`<div id="tag-menu" class="tag-menu-plain-text svelte-1p54cvv"><!></div>`);
var root_14 = from_html(`<div><!> <textarea data-testid="textbox" class="svelte-1p54cvv"></textarea> <!></div>`);
var root_17 = from_html(`<div class="submit-container svelte-1p54cvv"><button class="submit-button svelte-1p54cvv"><!></button></div>`);
var root = from_html(`<div><!> <!> <!> <!> <!></div>`);

function Dialogue($$anchor, $$props) {
	push($$props, true);
	const gradio = $$props.gradio;
	let checked = state(false);
	let disabled = user_derived(() => !gradio.shared.interactive);
	let dialogue_lines = state(proxy([]));

	user_effect(() => {
		if (gradio.props.value && gradio.props.value.length && typeof gradio.props.value !== "string") {
			set(dialogue_lines, [...gradio.props.value], true);
		} else if (gradio.props.value && typeof gradio.props.value !== "string") {
			set(
				dialogue_lines,
				[
					{
						speaker: `${gradio.props.speakers.length ? gradio.props.speakers[0] : ""}`,
						text: ""
					}
				],
				true
			);
		} else if (typeof gradio.props.value === "string") {
			set(textbox_value, gradio.props.value, true);
			set(checked, true);
		}
	});

	let buttons = user_derived(() => gradio.props.buttons || ["copy"]);

	let on_custom_button_click = (id) => {
		gradio.dispatch("custom_button_click", { id });
	};

	let old_value = state(proxy(gradio.props.value));

	user_effect(() => {
		if (get(old_value) != gradio.props.value) {
			set(old_value, gradio.props.value, true);
			gradio.dispatch("change");
		}
	});

	let dialogue_container_element;
	let showTagMenu = state(false);
	let currentLineIndex = state(-1);
	let selectedOptionIndex = state(0);
	let filtered_tags = state(proxy([]));
	let input_elements = proxy([]);
	let textarea_element;
	let offset_from_top = state(0);
	let copied = state(false);
	let timer;
	let textbox_value = state("");
	let hoveredSpeaker = state(null);
	let is_unformatting = state(false);
	let is_formatting = state(false);

	const defaultColorNames = [
		"red",
		"green",
		"blue",
		"yellow",
		"purple",
		"teal",
		"orange",
		"cyan",
		"lime",
		"pink"
	];

	const colorNameToHex = {
		red: "rgba(254, 202, 202, 0.7)",
		green: "rgba(209, 250, 229, 0.7)",
		blue: "rgba(219, 234, 254, 0.7)",
		yellow: "rgba(254, 243, 199, 0.7)",
		purple: "rgba(233, 213, 255, 0.7)",
		teal: "rgba(204, 251, 241, 0.7)",
		orange: "rgba(254, 215, 170, 0.7)",
		cyan: "rgba(207, 250, 254, 0.7)",
		lime: "rgba(217, 249, 157, 0.7)",
		pink: "rgba(252, 231, 243, 0.7)"
	};

	let speakerColors = user_derived(() => {
		let _speakerColors = {};

		if (gradio.props.color_map) {
			_speakerColors = { ...gradio.props.color_map };
		} else {
			_speakerColors = {};

			gradio.props.speakers.forEach((speaker, index) => {
				const colorName = defaultColorNames[index % defaultColorNames.length];

				_speakerColors[speaker] = colorNameToHex[colorName];
			});
		}

		return _speakerColors;
	});

	function add_line(index) {
		const newSpeaker = gradio.props.speakers.length > 0 ? gradio.props.speakers[0] : "";

		set(
			dialogue_lines,
			[
				...get(dialogue_lines).slice(0, index + 1),
				{ speaker: newSpeaker, text: "" },
				...get(dialogue_lines).slice(index + 1)
			],
			true
		);

		tick().then(() => {
			if (input_elements[index + 1]) {
				input_elements[index + 1].focus();
			}
		});

		gradio.props.value = [...get(dialogue_lines)];
	}

	function delete_line(index) {
		set(
			dialogue_lines,
			[
				...get(dialogue_lines).slice(0, index),
				...get(dialogue_lines).slice(index + 1)
			],
			true
		);

		gradio.props.value = [...get(dialogue_lines)];
	}

	function update_line(index, key, value) {
		get(dialogue_lines)[index][key] = value;
		set(dialogue_lines, [...get(dialogue_lines)], true);
		gradio.props.value = [...get(dialogue_lines)];
	}

	function handle_input(event, index) {
		const input = event.target || HTMLTextAreaElement;

		if (input && !input_elements[index]) {
			input_elements[index] = input;
		}

		const cursor_position = input.selectionStart || 0;
		const text = input.value;
		let show_menu = false;
		let position_reference_index = -1;

		if (text[cursor_position - 1] === ":") {
			set(currentLineIndex, index, true);
			position_reference_index = cursor_position;

			const search_text = get_tag_search_text(text, cursor_position);

			set(filtered_tags, gradio.props.tags.filter((tag) => search_text === "" || tag.toLowerCase().includes(search_text.toLowerCase())), true);
			show_menu = get(filtered_tags).length > 0;
			set(selectedOptionIndex, 0);
		} else {
			const lastColonIndex = text.lastIndexOf(":", cursor_position - 1);

			if (lastColonIndex >= 0 && !text.substring(lastColonIndex + 1, cursor_position).includes(" ")) {
				set(currentLineIndex, index, true);
				position_reference_index = lastColonIndex + 1; // Position menu relative to the start of the potential tag

				const searchText = text.substring(lastColonIndex + 1, cursor_position);

				set(filtered_tags, gradio.props.tags.filter((tag) => searchText === "" || tag.toLowerCase().includes(searchText.toLowerCase())), true);
				show_menu = get(filtered_tags).length > 0;
				set(selectedOptionIndex, 0);
			}
		}

		if (show_menu && position_reference_index !== -1) {
			set(showTagMenu, true);

			const input_rect = input.getBoundingClientRect();

			// Position menu below the current input by calculating the distance from the top of the container
			// and use 1.5 times the input height.
			if (dialogue_container_element) {
				const container_rect = dialogue_container_element.getBoundingClientRect();

				set(offset_from_top, container_rect.top + input_rect.height * (index + 1.5));
			}
		} else {
			set(showTagMenu, false);
		}

		gradio.dispatch("input");
	}

	function get_tag_search_text(text, cursorPosition) {
		const lastColonIndex = text.lastIndexOf(":", cursorPosition - 1);

		if (lastColonIndex >= 0) {
			return text.substring(lastColonIndex + 1, cursorPosition);
		}

		return "";
	}

	async function insert_selected_tag() {
		const tag = get(filtered_tags)[get(selectedOptionIndex)];

		if (tag) {
			let text;
			let currentInput;

			if (get(checked)) {
				currentInput = textarea_element;
				text = get(textbox_value);
			} else {
				currentInput = input_elements[get(currentLineIndex)];
				text = get(dialogue_lines)[get(currentLineIndex)].text;
			}

			const cursorPosition = currentInput?.selectionStart || 0;
			const lastColonIndex = text.lastIndexOf(":", cursorPosition - 1);

			if (lastColonIndex >= 0) {
				const beforeColon = text.substring(0, lastColonIndex);
				const afterCursor = text.substring(cursorPosition);

				if (get(checked)) {
					// plain text mode: don't filter speaker tags
					const newText = `${beforeColon}${tag} ${afterCursor}`;

					set(textbox_value, newText);

					if (gradio.props.speakers.length === 0) {
						gradio.props.value = newText;
					} else {
						gradio.props.value = await gradio.shared.server.unformat({ text: newText });
					}

					tick().then(() => {
						if (textarea_element) {
							const newCursorPosition = beforeColon.length + tag.length + 1;

							textarea_element.setSelectionRange(newCursorPosition, newCursorPosition);
							textarea_element.focus();
						}
					});
				} else {
					// dialogue line mode
					const filteredBeforeColon = beforeColon.replace(/\[S\d+\]/g, "").trim();

					const newText = `${filteredBeforeColon}${tag} ${afterCursor}`;

					update_line(get(currentLineIndex), "text", newText);

					tick().then(() => {
						const updatedInput = input_elements[get(currentLineIndex)];

						if (updatedInput) {
							const newCursorPosition = filteredBeforeColon.length + tag.length + 1;

							updatedInput.setSelectionRange(newCursorPosition, newCursorPosition);
							updatedInput.focus();
						}
					});
				}
			}

			set(showTagMenu, false);
			set(selectedOptionIndex, 0);
		}
	}

	async function insert_tag(index) {
		index = parseInt(index);

		const tag = gradio.props.tags[index];

		if (tag) {
			let text;
			let currentInput;

			if (get(checked)) {
				currentInput = textarea_element;
				text = get(textbox_value);
			} else {
				currentInput = input_elements[get(currentLineIndex)];
				text = get(dialogue_lines)[get(currentLineIndex)].text;
			}

			const cursorPosition = currentInput?.selectionStart || 0;
			const lastColonIndex = text.lastIndexOf(":", cursorPosition - 1);

			if (lastColonIndex >= 0) {
				const beforeColon = text.substring(0, lastColonIndex);
				const afterCursor = text.substring(cursorPosition);

				if (get(checked)) {
					// plain text mode: don't filter speaker tags
					const newText = `${beforeColon}${tag} ${afterCursor}`;

					set(textbox_value, newText);

					if (gradio.props.speakers.length === 0) {
						gradio.props.value = newText;
					} else {
						gradio.props.value = await gradio.shared.server.unformat({ text: newText });
					}

					tick().then(() => {
						if (textarea_element) {
							const newCursorPosition = beforeColon.length + tag.length + 1;

							textarea_element.setSelectionRange(newCursorPosition, newCursorPosition);
							textarea_element.focus();
						}
					});
				} else {
					// dialogue line mode
					const filteredBeforeColon = beforeColon.replace(/\[S\d+\]/g, "").trim();

					const newText = `${filteredBeforeColon}${tag} ${afterCursor}`;

					update_line(get(currentLineIndex), "text", newText);

					tick().then(() => {
						const updatedInput = input_elements[get(currentLineIndex)];

						if (updatedInput) {
							const newCursorPosition = filteredBeforeColon.length + tag.length + 1;

							updatedInput.setSelectionRange(newCursorPosition, newCursorPosition);
							updatedInput.focus();
						}
					});
				}
			}

			set(showTagMenu, false);
			set(selectedOptionIndex, 0);
		}
	}

	function handle_click_outside(event) {
		if (get(showTagMenu)) {
			const target = event.target;
			const tagMenu = document.getElementById("tag-menu");

			if (tagMenu && !tagMenu.contains(target)) {
				set(showTagMenu, false);
			}
		}
	}

	async function value_to_string(value) {
		if (typeof value === "string") {
			return value;
		}

		return await gradio.shared.server.format(value);
	}

	async function handle_copy() {
		if ("clipboard" in navigator) {
			const text = await value_to_string(gradio.props.value);

			await navigator.clipboard.writeText(text);
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

	async function handle_submit() {
		if (get(checked)) {
			gradio.props.value = await gradio.shared.server.unformat({ text: get(textbox_value) });
		}

		gradio.dispatch("submit");
	}

	onMount(async () => {
		if (typeof gradio.props.value === "string") {
			set(textbox_value, gradio.props.value, true);
		} else if (gradio.props.value && gradio.props.value.length > 0) {
			const formatted = await value_to_string(gradio.props.value);

			set(textbox_value, formatted, true);
		} else {
			set(textbox_value, "");
		}
	});

	var div = root();

	event('click', $window, handle_click_outside);

	let classes;
	var node = child(div);

	{
		var consequent_1 = ($$anchor) => {
			IconButtonWrapper($$anchor, {
				get buttons() {
					return get(buttons);
				},
				on_custom_button_click,
				children: ($$anchor, $$slotProps) => {
					var fragment_1 = comment();
					var node_1 = first_child(fragment_1);

					{
						var consequent = ($$anchor) => {
							{
								let $0 = user_derived(() => get(copied) ? Check : Copy);
								let $1 = user_derived(() => get(copied) ? "Copied" : "Copy");

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
						};

						if_block(node_1, ($$render) => {
							if (get(buttons).some((btn) => typeof btn === "string" && btn === "copy")) $$render(consequent);
						});
					}

					append($$anchor, fragment_1);
				},
				$$slots: { default: true }
			});
		};

		if_block(node, ($$render) => {
			if (gradio.shared.show_label && (get(buttons).some((btn) => typeof btn === "string" && btn === "copy") || get(buttons).some((btn) => typeof btn !== "string"))) $$render(consequent_1);
		});
	}

	var node_2 = sibling(node, 2);

	BlockTitle(node_2, {
		get show_label() {
			return gradio.shared.show_label;
		},

		get info() {
			return gradio.props.info;
		},

		children: ($$anchor, $$slotProps) => {
			next();

			var text_1 = text();

			template_effect(() => set_text(text_1, gradio.shared.label || "Dialogue"));
			append($$anchor, text_1);
		},
		$$slots: { default: true }
	});

	var node_3 = sibling(node_2, 2);

	{
		var consequent_2 = ($$anchor) => {
			var div_1 = root_5();
			let classes_1;
			var node_4 = child(div_1);

			{
				let $0 = user_derived(() => get(is_formatting) || get(is_unformatting));

				Switch(node_4, {
					label: 'Plain Text',
					get disabled() {
						return get($0);
					},

					get checked() {
						return get(checked);
					},

					set checked($$value) {
						set(checked, $$value, true);
					},

					$$events: {
						click: async (e) => {
							if (!e.detail.checked) {
								set(is_unformatting, true);

								try {
									gradio.props.value = await gradio.shared.server.unformat({ text: get(textbox_value) });
									set(dialogue_lines, [...gradio.props.value], true);
								} finally {
									set(is_unformatting, false);
								}
							} else {
								set(is_formatting, true);

								try {
									set(textbox_value, await value_to_string(get(dialogue_lines)), true);
								} finally {
									set(is_formatting, false);
								}
							}
						}
					}
				});
			}

			reset(div_1);

			template_effect(() => classes_1 = set_class(div_1, 1, 'switch-container top-switch svelte-1p54cvv', null, classes_1, {
				'switch-disabled': get(is_formatting) || get(is_unformatting)
			}));

			append($$anchor, div_1);
		};

		if_block(node_3, ($$render) => {
			if (gradio.props.ui_mode === "both") $$render(consequent_2);
		});
	}

	var node_5 = sibling(node_3, 2);

	{
		var consequent_7 = ($$anchor) => {
			var div_2 = root_6();
			let classes_2;
			var node_6 = child(div_2);

			{
				var consequent_3 = ($$anchor) => {
					var div_3 = root_7();

					transition(3, div_3, () => fade, () => ({ duration: 200 }));
					append($$anchor, div_3);
				};

				if_block(node_6, ($$render) => {
					if (get(is_unformatting)) $$render(consequent_3);
				});
			}

			var node_7 = sibling(node_6, 2);

			each(node_7, 17, () => get(dialogue_lines), index, ($$anchor, line, i) => {
				var div_4 = root_8();
				var div_5 = child(div_4);
				var node_8 = child(div_5);

				{
					var consequent_4 = ($$anchor) => {
						var textarea = root_9();

						remove_textarea_child(textarea);
						template_effect(() => textarea.disabled = get(disabled));
						bind_value(textarea, () => get(line).speaker, ($$value) => (get(line).speaker = $$value));
						append($$anchor, textarea);
					};

					var alternate = ($$anchor) => {
						{
							let $0 = user_derived(() => gradio.props.speakers.map((s) => [s, s]));

							Dropdown($$anchor, {
								label: '',
								show_label: false,
								container: true,
								interactive: true,
								get value() {
									return get(line).speaker;
								},

								get choices() {
									return get($0);
								},
								on_change: (val) => update_line(i, "speaker", val)
							});
						}
					};

					if_block(node_8, ($$render) => {
						if (get(disabled)) $$render(consequent_4); else $$render(alternate, false);
					});
				}

				reset(div_5);

				var div_6 = sibling(div_5, 2);
				var div_7 = child(div_6);
				var textarea_1 = child(div_7);

				remove_textarea_child(textarea_1);
				bind_this(textarea_1, ($$value, i) => input_elements[i] = $$value, (i) => input_elements?.[i], () => [i]);

				var node_9 = sibling(textarea_1, 2);

				{
					var consequent_5 = ($$anchor) => {
						var div_8 = root_11();
						var node_10 = child(div_8);

						{
							let $0 = user_derived(() => gradio.props.tags.map((s, i) => [s, i]));
							let $1 = user_derived(() => get(filtered_tags).map((s) => gradio.props.tags.indexOf(s)));
							let $2 = user_derived(() => get(filtered_tags).map((s) => gradio.props.tags.indexOf(s))[get(selectedOptionIndex)]);

							DropdownOptions(node_10, {
								get choices() {
									return get($0);
								},

								get filtered_indices() {
									return get($1);
								},

								get active_index() {
									return get($2);
								},
								show_options: true,
								onchange: (e) => insert_tag(e),
								get offset_from_top() {
									return get(offset_from_top);
								},
								from_top: true
							});
						}

						reset(div_8);
						transition(3, div_8, () => fade, () => ({ duration: 100 }));
						append($$anchor, div_8);
					};

					if_block(node_9, ($$render) => {
						if (get(showTagMenu) && get(currentLineIndex) === i) $$render(consequent_5);
					});
				}

				reset(div_7);
				reset(div_6);

				var node_11 = sibling(div_6, 2);

				{
					var consequent_6 = ($$anchor) => {
						var div_9 = root_12();
						let classes_3;
						var button = child(div_9);
						var node_12 = child(button);

						Plus(node_12);
						reset(button);
						reset(div_9);

						template_effect(() => {
							classes_3 = set_class(div_9, 1, 'svelte-1p54cvv', null, classes_3, { 'action-column': i == 0, hidden: get(disabled) });
							button.disabled = get(disabled);
						});

						event('click', button, () => add_line(i));
						append($$anchor, div_9);
					};

					if_block(node_11, ($$render) => {
						if (gradio.props.max_lines == undefined || gradio.props.max_lines && i < gradio.props.max_lines - 1) $$render(consequent_6);
					});
				}

				var div_10 = sibling(node_11, 2);
				let classes_4;
				var button_1 = child(div_10);
				var node_13 = child(button_1);

				Trash(node_13);
				reset(button_1);
				reset(div_10);
				reset(div_4);

				template_effect(() => {
					set_style(div_4, `--speaker-bg-color: ${(get(disabled) && (get(hoveredSpeaker) === null || get(hoveredSpeaker) === get(line).speaker)
						? get(speakerColors)[get(line).speaker] || 'transparent'
						: 'transparent') ?? ''}`);

					set_attribute(textarea_1, 'placeholder', gradio.props.placeholder);
					textarea_1.disabled = get(disabled);
					classes_4 = set_class(div_10, 1, 'action-column svelte-1p54cvv', null, classes_4, { hidden: get(disabled) || i == 0 });
					button_1.disabled = get(disabled);
				});

				event('mouseenter', div_5, () => get(disabled) && set(hoveredSpeaker, get(line).speaker, true));
				event('mouseleave', div_5, () => get(disabled) && set(hoveredSpeaker, null));
				bind_value(textarea_1, () => get(line).text, ($$value) => (get(line).text = $$value));
				event('input', textarea_1, (event) => handle_input(event, i));
				event('focus', textarea_1, (event) => handle_input(event, i));

				event('keydown', textarea_1, (event) => {
					if (event.key === "Escape" && get(showTagMenu)) {
						set(showTagMenu, false);
						set(selectedOptionIndex, 0);
						event.preventDefault();
					} else if (get(showTagMenu) && get(currentLineIndex) === i) {
						if (event.key === "ArrowDown") {
							set(selectedOptionIndex, Math.min(get(selectedOptionIndex) + 1, get(filtered_tags).length - 1), true);
							event.preventDefault();
						} else if (event.key === "ArrowUp") {
							set(selectedOptionIndex, Math.max(get(selectedOptionIndex) - 1, 0), true);
							event.preventDefault();
						} else if (event.key === "Enter") {
							if (get(filtered_tags)[get(selectedOptionIndex)]) {
								insert_selected_tag();
							}

							event.preventDefault();
						}
					}
				});

				event('click', button_1, () => delete_line(i));
				append($$anchor, div_4);
			});

			reset(div_2);
			bind_this(div_2, ($$value) => dialogue_container_element = $$value, () => dialogue_container_element);
			template_effect(() => classes_2 = set_class(div_2, 1, 'dialogue-container svelte-1p54cvv', null, classes_2, { loading: get(is_unformatting) }));
			append($$anchor, div_2);
		};

		var alternate_1 = ($$anchor) => {
			var fragment_5 = comment();
			var node_14 = first_child(fragment_5);

			{
				var consequent_10 = ($$anchor) => {
					var div_11 = root_14();
					let classes_5;
					var node_15 = child(div_11);

					{
						var consequent_8 = ($$anchor) => {
							var div_12 = root_15();

							transition(3, div_12, () => fade, () => ({ duration: 200 }));
							append($$anchor, div_12);
						};

						if_block(node_15, ($$render) => {
							if (get(is_formatting)) $$render(consequent_8);
						});
					}

					var textarea_2 = sibling(node_15, 2);

					remove_textarea_child(textarea_2);
					set_attribute(textarea_2, 'rows', 5);
					bind_this(textarea_2, ($$value) => textarea_element = $$value, () => textarea_element);

					var node_16 = sibling(textarea_2, 2);

					{
						var consequent_9 = ($$anchor) => {
							var div_13 = root_16();
							var node_17 = child(div_13);

							{
								let $0 = user_derived(() => gradio.props.tags.map((s, i) => [s, i]));
								let $1 = user_derived(() => get(filtered_tags).map((s) => gradio.props.tags.indexOf(s)));
								let $2 = user_derived(() => get(filtered_tags).map((s) => gradio.props.tags.indexOf(s))[get(selectedOptionIndex)]);

								DropdownOptions(node_17, {
									get choices() {
										return get($0);
									},

									get filtered_indices() {
										return get($1);
									},

									get active_index() {
										return get($2);
									},
									show_options: true,
									$$events: { change: (e) => insert_tag(e) }
								});
							}

							reset(div_13);
							transition(3, div_13, () => fade, () => ({ duration: 100 }));
							append($$anchor, div_13);
						};

						if_block(node_16, ($$render) => {
							if (get(showTagMenu)) $$render(consequent_9);
						});
					}

					reset(div_11);

					template_effect(() => {
						classes_5 = set_class(div_11, 1, 'textarea-container svelte-1p54cvv', null, classes_5, { loading: get(is_formatting) });
						set_attribute(textarea_2, 'placeholder', gradio.props.placeholder);
						textarea_2.disabled = get(disabled);
					});

					bind_value(textarea_2, () => get(textbox_value), ($$value) => set(textbox_value, $$value));

					event('input', textarea_2, (event) => {
						handle_input(event, 0);
						gradio.props.value = get(textbox_value);
					});

					event('focus', textarea_2, (event) => handle_input(event, 0));

					event('keydown', textarea_2, (event) => {
						if (event.key === "Escape" && get(showTagMenu)) {
							set(showTagMenu, false);
							set(selectedOptionIndex, 0);
							event.preventDefault();
						} else if (get(showTagMenu)) {
							if (event.key === "ArrowDown") {
								set(selectedOptionIndex, Math.min(get(selectedOptionIndex) + 1, get(filtered_tags).length - 1), true);
								event.preventDefault();
							} else if (event.key === "ArrowUp") {
								set(selectedOptionIndex, Math.max(get(selectedOptionIndex) - 1, 0), true);
								event.preventDefault();
							} else if (event.key === "Enter") {
								if (get(filtered_tags)[get(selectedOptionIndex)]) {
									insert_selected_tag();
								}

								event.preventDefault();
							}
						}
					});

					append($$anchor, div_11);
				};

				if_block(
					node_14,
					($$render) => {
						if (get(checked) || gradio.props.ui_mode !== "dialogue") $$render(consequent_10);
					},
					true
				);
			}

			append($$anchor, fragment_5);
		};

		if_block(node_5, ($$render) => {
			if (!get(checked) && gradio.props.ui_mode !== "text") $$render(consequent_7); else $$render(alternate_1, false);
		});
	}

	var node_18 = sibling(node_5, 2);

	{
		var consequent_12 = ($$anchor) => {
			var div_14 = root_17();
			var button_2 = child(div_14);
			var node_19 = child(button_2);

			{
				var consequent_11 = ($$anchor) => {
					var text_2 = text();

					template_effect(() => set_text(text_2, gradio.props.submit_btn));
					append($$anchor, text_2);
				};

				var alternate_2 = ($$anchor) => {
					Send($$anchor);
				};

				if_block(node_19, ($$render) => {
					if (typeof gradio.props.submit_btn === "string") $$render(consequent_11); else $$render(alternate_2, false);
				});
			}

			reset(button_2);
			reset(div_14);
			template_effect(() => button_2.disabled = get(disabled));
			event('click', button_2, handle_submit);
			append($$anchor, div_14);
		};

		if_block(node_18, ($$render) => {
			if (gradio.props.submit_btn && !get(disabled)) $$render(consequent_12);
		});
	}

	reset(div);
	template_effect(() => classes = set_class(div, 1, 'svelte-1p54cvv', null, classes, { container: gradio.shared.container }));
	append($$anchor, div);
	pop();
}

var root_1 = from_html(`<!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);

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

		get scale() {
			return gradio.shared.scale;
		},

		get min_width() {
			return gradio.shared.min_width;
		},
		allow_overflow: true,
		get padding() {
			return gradio.shared.container;
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
						() => gradio.shared.loading_status,
						{
							on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
						}
					));
				};

				if_block(node, ($$render) => {
					if (gradio.shared.loading_status) $$render(consequent);
				});
			}

			var node_1 = sibling(node, 2);

			Dialogue(node_1, {
				get gradio() {
					return gradio;
				}
			});

			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

export { Dialogue as BaseDialogue, Index as default };
//# sourceMappingURL=Index-BAJ5KXXp.js.map
