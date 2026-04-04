import { p as prop, i as if_block, g as spread_props, r as rest_props } from './i18n-CMo5lpzy.js';
import { R as push, ae as onDestroy, w as get, T as pop, a5 as user_derived, x as set, u as state, S as first_child, a as append, X as sibling, W as from_html, v as proxy, y as user_effect } from './index-BvBk1Iap.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import { C as Code$1 } from './Code-C9UfsaaR.js';
import { C as Check } from './Check-BlnBgjLT.js';
import { C as Copy } from './Copy-DFFray7x.js';
import { I as IconButton } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { D as Download } from './Download-pFhUaTl4.js';
import { D as DownloadLink } from './DownloadLink-KvrFKyMn.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { B as BlockLabel } from './BlockLabel-oGSqUii_.js';
import { E as Empty } from './Empty-CHa-gxv7.js';
import { C as Code } from './Code-DMRxApdQ.js';
export { default as BaseExample } from './Example-obvXP67p.js';
import './clone-DaB-S2nH.js';
import './Clear-Dgs5UhTO.js';
import './html-CdUEtR5E.js';
import './index-C7e-J7CF.js';
import './snippet-Bl5YVoCg.js';
/* empty css                                               */

function Copy_1($$anchor, $$props) {
	push($$props, true);

	let copied = state(false);
	let timer;

	function copy_feedback() {
		set(copied, true);

		if (timer) clearTimeout(timer);

		timer = setTimeout(
			() => {
				set(copied, false);
			},
			2000
		);
	}

	async function handle_copy() {
		if ("clipboard" in navigator) {
			await navigator.clipboard.writeText($$props.value);
			copy_feedback();
		}
	}

	onDestroy(() => {
		if (timer) clearTimeout(timer);
	});

	{
		let $0 = user_derived(() => get(copied) ? Check : Copy);

		IconButton($$anchor, {
			get Icon() {
				return get($0);
			},
			onclick: handle_copy
		});
	}

	pop();
}

function Download_1($$anchor, $$props) {
	push($$props, true);

	let ext = user_derived(() => get_ext_for_type($$props.language));

	function get_ext_for_type(type) {
		const exts = {
			py: "py",
			python: "py",
			md: "md",
			markdown: "md",
			json: "json",
			html: "html",
			css: "css",
			js: "js",
			javascript: "js",
			ts: "ts",
			typescript: "ts",
			yaml: "yaml",
			yml: "yml",
			dockerfile: "dockerfile",
			sh: "sh",
			shell: "sh",
			r: "r",
			c: "c",
			cpp: "cpp",
			latex: "tex"
		};

		return exts[type] || "txt";
	}

	let copied = state(false);
	let timer;

	function copy_feedback() {
		set(copied, true);

		if (timer) clearTimeout(timer);

		timer = setTimeout(
			() => {
				set(copied, false);
			},
			2000
		);
	}

	let download_value = user_derived(() => URL.createObjectURL(new Blob([$$props.value])));

	onDestroy(() => {
		if (timer) clearTimeout(timer);
	});

	DownloadLink($$anchor, {
		get download() {
			return `file.${get(ext) ?? ''}`;
		},

		get href() {
			return get(download_value);
		},
		onclick: copy_feedback,
		children: ($$anchor, $$slotProps) => {
			{
				let $0 = user_derived(() => get(copied) ? Check : Download);

				IconButton($$anchor, {
					get Icon() {
						return get($0);
					}
				});
			}
		},
		$$slots: { default: true }
	});

	pop();
}

var root_1$1 = from_html(`<!> <!>`, 1);

function Widgets($$anchor, $$props) {
	push($$props, true);

	let buttons = prop($$props, 'buttons', 3, null),
		on_custom_button_click = prop($$props, 'on_custom_button_click', 3, null);

	IconButtonWrapper($$anchor, {
		get buttons() {
			return buttons();
		},

		get on_custom_button_click() {
			return on_custom_button_click();
		},

		children: ($$anchor, $$slotProps) => {
			var fragment_1 = root_1$1();
			var node = first_child(fragment_1);

			{
				var consequent = ($$anchor) => {
					Download_1($$anchor, {
						get value() {
							return $$props.value;
						},

						get language() {
							return $$props.language;
						}
					});
				};

				if_block(node, ($$render) => {
					if (buttons()?.some((btn) => typeof btn === "string" && btn === "download")) $$render(consequent);
				});
			}

			var node_1 = sibling(node, 2);

			{
				var consequent_1 = ($$anchor) => {
					Copy_1($$anchor, {
						get value() {
							return $$props.value;
						}
					});
				};

				if_block(node_1, ($$render) => {
					if (buttons()?.some((btn) => typeof btn === "string" && btn === "copy")) $$render(consequent_1);
				});
			}

			append($$anchor, fragment_1);
		},
		$$slots: { default: true }
	});

	pop();
}

var root_5 = from_html(`<!> <!>`, 1);
var root_1 = from_html(`<!> <!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new Gradio(props);
	let dark_mode = gradio.shared.theme === "dark";
	let label = user_derived(() => gradio.shared.label || gradio.i18n("code.code"));
	let old_value = state(proxy(gradio.props.value));
	let first_change = true;

	user_effect(() => {
		if (first_change) {
			first_change = false;

			return;
		}

		if (get(old_value) != gradio.props.value) {
			set(old_value, gradio.props.value, true);
			gradio.dispatch("change");
		}
	});

	{
		let $0 = user_derived(() => gradio.props.max_lines && "fit-content");

		Block($$anchor, {
			get height() {
				return get($0);
			},
			variant: "solid",
			padding: false,
			get elem_id() {
				return gradio.shared.elem_id;
			},

			get elem_classes() {
				return gradio.shared.elem_classes;
			},

			get visible() {
				return gradio.shared.visible;
			},

			get scale() {
				return gradio.shared.scale;
			},

			get min_width() {
				return gradio.shared.min_width;
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
						on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
					}
				));

				var node_1 = sibling(node, 2);

				{
					var consequent = ($$anchor) => {
						BlockLabel($$anchor, {
							get Icon() {
								return Code;
							},

							get show_label() {
								return gradio.shared.show_label;
							},

							get label() {
								return get(label);
							},
							float: false
						});
					};

					if_block(node_1, ($$render) => {
						if (gradio.shared.show_label) $$render(consequent);
					});
				}

				var node_2 = sibling(node_1, 2);

				{
					var consequent_1 = ($$anchor) => {
						Empty($$anchor, {
							unpadded_box: true,
							size: 'large',
							children: ($$anchor, $$slotProps) => {
								Code($$anchor);
							},
							$$slots: { default: true }
						});
					};

					var alternate = ($$anchor) => {
						var fragment_5 = root_5();
						var node_3 = first_child(fragment_5);

						{
							let $0 = user_derived(() => gradio.props.buttons ?? ["copy", "download"]);

							Widgets(node_3, {
								get language() {
									return gradio.props.language;
								},

								get value() {
									return gradio.props.value;
								},

								get buttons() {
									return get($0);
								},

								on_custom_button_click: (id) => {
									gradio.dispatch("custom_button_click", { id });
								}
							});
						}

						var node_4 = sibling(node_3, 2);

						{
							let $0 = user_derived(() => !gradio.shared.interactive);

							Code$1(node_4, {
								get language() {
									return gradio.props.language;
								},

								get lines() {
									return gradio.props.lines;
								},

								get max_lines() {
									return gradio.props.max_lines;
								},

								get dark_mode() {
									return dark_mode;
								},

								get wrap_lines() {
									return gradio.props.wrap_lines;
								},

								get show_line_numbers() {
									return gradio.props.show_line_numbers;
								},

								get autocomplete() {
									return gradio.props.autocomplete;
								},

								get readonly() {
									return get($0);
								},
								onblur: () => gradio.dispatch("blur"),
								onfocus: () => gradio.dispatch("focus"),
								oninput: () => gradio.dispatch("input"),
								get value() {
									return gradio.props.value;
								},

								set value($$value) {
									gradio.props.value = $$value;
								}
							});
						}

						append($$anchor, fragment_5);
					};

					if_block(node_2, ($$render) => {
						if (!gradio.props.value && !gradio.shared.interactive) $$render(consequent_1); else $$render(alternate, false);
					});
				}

				append($$anchor, fragment_1);
			},
			$$slots: { default: true }
		});
	}

	pop();
}

export { Code$1 as BaseCode, Copy_1 as BaseCopy, Download_1 as BaseDownload, Widgets as BaseWidget, Index as default };
//# sourceMappingURL=Index-BuAoQXgO.js.map
