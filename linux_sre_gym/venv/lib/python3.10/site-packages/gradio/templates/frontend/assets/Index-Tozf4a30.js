import { p as prop, i as if_block, g as spread_props, r as rest_props } from './i18n-CMo5lpzy.js';
import { R as push, S as first_child, w as get, a as append, T as pop, X as sibling, a5 as user_derived, W as from_html, u as state, v as proxy, y as user_effect, x as set, M as tick } from './index-BvBk1Iap.js';
import { s as snapshot } from './clone-DaB-S2nH.js';
import { G as Gradio } from './utils.svelte-CYLB_jIA.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { B as BlockLabel } from './BlockLabel-oGSqUii_.js';
import { E as Empty } from './Empty-CHa-gxv7.js';
import { F as File } from './File-_cC4Z_uD.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { F as FilePreview, a as FileUpload } from './FileUpload-9lL-iwhh.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { U as UploadText } from './UploadText-BSD1IoCI.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
export { default as BaseExample } from './Example-CI1jJDZu.js';
import './snippet-Bl5YVoCg.js';
import './Upload-CPeZiqam.js';
import './actions-Cit_UA5S.js';
import './Clear-Dgs5UhTO.js';
import './html-CdUEtR5E.js';
import './DownloadLink-KvrFKyMn.js';
import './index-C7e-J7CF.js';
/* empty css                                               */

var root = from_html(`<!> <!> <!>`, 1);

function File_1($$anchor, $$props) {
	push($$props, true);

	let buttons = prop($$props, 'buttons', 3, null),
		on_custom_button_click = prop($$props, 'on_custom_button_click', 3, null);

	var fragment = root();
	var node = first_child(fragment);

	{
		var consequent = ($$anchor) => {
			IconButtonWrapper($$anchor, {
				get buttons() {
					return buttons();
				},

				get on_custom_button_click() {
					return on_custom_button_click();
				}
			});
		};

		if_block(node, ($$render) => {
			if ($$props.show_label && buttons() && buttons().length > 0) $$render(consequent);
		});
	}

	var node_1 = sibling(node, 2);

	{
		let $0 = user_derived(() => $$props.value === null);
		let $1 = user_derived(() => $$props.label || "File");

		BlockLabel(node_1, {
			get show_label() {
				return $$props.show_label;
			},

			get float() {
				return get($0);
			},

			get Icon() {
				return File;
			},

			get label() {
				return get($1);
			}
		});
	}

	var node_2 = sibling(node_1, 2);

	{
		var consequent_1 = ($$anchor) => {
			FilePreview($$anchor, {
				get i18n() {
					return $$props.i18n;
				},

				get selectable() {
					return $$props.selectable;
				},

				get value() {
					return $$props.value;
				},

				get height() {
					return $$props.height;
				},

				$$events: {
					select(...$$args) {
						$$props.on_select?.apply(this, $$args);
					},

					download(...$$args) {
						$$props.on_download?.apply(this, $$args);
					}
				}
			});
		};

		var alternate = ($$anchor) => {
			Empty($$anchor, {
				unpadded_box: true,
				size: 'large',
				children: ($$anchor, $$slotProps) => {
					File($$anchor);
				},
				$$slots: { default: true }
			});
		};

		if_block(node_2, ($$render) => {
			if ($$props.value && (Array.isArray($$props.value) ? $$props.value.length > 0 : true)) $$render(consequent_1); else $$render(alternate, false);
		});
	}

	append($$anchor, fragment);
	pop();
}

var root_1 = from_html(`<!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	const props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	let upload_promise = state(null);
	let dragging = state(false);

	class FileGradio extends Gradio {
		async get_data() {
			if (get(upload_promise)) {
				await get(upload_promise);
				await tick();
			}

			const data = await super.get_data();

			return data;
		}
	}

	const gradio = new FileGradio(props);
	let old_value = state(proxy(gradio.props.value));

	user_effect(() => {
		if (get(old_value) !== gradio.props.value) {
			set(old_value, gradio.props.value, true);
			gradio.dispatch("change", snapshot(gradio.props.value));
		}
	});

	{
		let $0 = user_derived(() => gradio.props.value ? "solid" : "dashed");
		let $1 = user_derived(() => get(dragging) ? "focus" : "base");

		Block($$anchor, {
			get visible() {
				return gradio.shared.visible;
			},

			get variant() {
				return get($0);
			},

			get border_mode() {
				return get($1);
			},
			padding: false,
			get elem_id() {
				return gradio.shared.elem_id;
			},

			get elem_classes() {
				return gradio.shared.elem_classes;
			},

			get container() {
				return gradio.shared.container;
			},

			get scale() {
				return gradio.shared.scale;
			},

			get min_width() {
				return gradio.shared.min_width;
			},
			allow_overflow: false,
			children: ($$anchor, $$slotProps) => {
				var fragment_1 = root_1();
				var node = first_child(fragment_1);

				{
					let $0 = user_derived(() => gradio.shared.loading_status?.status || "complete");

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
							get status() {
								return get($0);
							},
							on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
						}
					));
				}

				var node_1 = sibling(node, 2);

				{
					var consequent = ($$anchor) => {
						File_1($$anchor, {
							on_select: ({ detail }) => gradio.dispatch("select", detail),
							on_download: ({ detail }) => gradio.dispatch("download", detail),
							get selectable() {
								return gradio.props._selectable;
							},

							get value() {
								return gradio.props.value;
							},

							get label() {
								return gradio.shared.label;
							},

							get show_label() {
								return gradio.shared.show_label;
							},

							get height() {
								return gradio.props.height;
							},

							get i18n() {
								return gradio.i18n;
							},

							get buttons() {
								return gradio.props.buttons;
							},

							on_custom_button_click: (id) => {
								gradio.dispatch("custom_button_click", { id });
							}
						});
					};

					var alternate = ($$anchor) => {
						{
							let $0 = user_derived(() => gradio.props.height ?? undefined);

							FileUpload($$anchor, {
								upload: (...args) => gradio.shared.client.upload(...args),
								stream_handler: (...args) => gradio.shared.client.stream(...args),
								get label() {
									return gradio.shared.label;
								},

								get show_label() {
									return gradio.shared.show_label;
								},

								get value() {
									return gradio.props.value;
								},

								get file_count() {
									return gradio.props.file_count;
								},

								get file_types() {
									return gradio.props.file_types;
								},

								get selectable() {
									return gradio.props._selectable;
								},

								get height() {
									return get($0);
								},

								get root() {
									return gradio.shared.root;
								},

								get allow_reordering() {
									return gradio.props.allow_reordering;
								},

								get max_file_size() {
									return gradio.shared.max_file_size;
								},

								get buttons() {
									return gradio.props.buttons;
								},

								on_custom_button_click: (id) => {
									gradio.dispatch("custom_button_click", { id });
								},

								onchange: (detail) => {
									gradio.props.value = detail;
								},
								ondrag: (detail) => set(dragging, detail, true),
								onclear: () => gradio.dispatch("clear"),
								onselect: (detail) => gradio.dispatch("select", detail),
								onupload: () => gradio.dispatch("upload"),
								onerror: (error) => {
									gradio.shared.loading_status = gradio.shared.loading_status || {};
									gradio.shared.loading_status.status = "error";
									gradio.dispatch("error", error);
								},

								ondelete: (detail) => {
									gradio.dispatch("delete", detail);
								},

								get i18n() {
									return gradio.i18n;
								},

								get upload_promise() {
									return get(upload_promise);
								},

								set upload_promise($$value) {
									set(upload_promise, $$value, true);
								},

								children: ($$anchor, $$slotProps) => {
									UploadText($$anchor, {
										get i18n() {
											return gradio.i18n;
										},
										type: 'file'
									});
								},
								$$slots: { default: true }
							});
						}
					};

					if_block(node_1, ($$render) => {
						if (!gradio.shared.interactive) $$render(consequent); else $$render(alternate, false);
					});
				}

				append($$anchor, fragment_1);
			},
			$$slots: { default: true }
		});
	}

	pop();
}

export { File_1 as BaseFile, FileUpload as BaseFileUpload, FilePreview, Index as default };
//# sourceMappingURL=Index-Tozf4a30.js.map
