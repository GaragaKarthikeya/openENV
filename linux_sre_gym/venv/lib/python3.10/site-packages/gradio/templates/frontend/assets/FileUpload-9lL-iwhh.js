import { p as prop, k as each, i as if_block, b as set_class, a as set_attribute, f as set_style, s as slot } from './i18n-CMo5lpzy.js';
import { R as push, t as template_effect, a as append, T as pop, w as get, W as from_html, a5 as user_derived, V as child, aj as delegate, X as sibling, a0 as set_text, Z as event, S as first_child, a7 as text, u as state, x as set, Y as reset, a8 as next, a6 as comment, M as tick } from './index-BvBk1Iap.js';
import { U as Upload } from './Upload-CPeZiqam.js';
import { I as IconButton, U as Upload$1 } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { B as BlockLabel } from './BlockLabel-oGSqUii_.js';
import { C as Clear } from './Clear-Dgs5UhTO.js';
import { F as File } from './File-_cC4Z_uD.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { h as html } from './html-CdUEtR5E.js';
import { D as DownloadLink } from './DownloadLink-KvrFKyMn.js';

const prettyBytes = (bytes) => {
  let units = ["B", "KB", "MB", "GB", "PB"];
  let i = 0;
  while (bytes > 1024) {
    bytes /= 1024;
    i++;
  }
  let unit = units[i];
  return bytes.toFixed(1) + "&nbsp;" + unit;
};

var root_2 = from_html(`<span class="drag-handle svelte-al0bnp">⋮⋮</span>`);
var root_4$1 = from_html(`<!>&nbsp;&#8675;`, 1);
var root_6 = from_html(`<td class="svelte-al0bnp"><button class="label-clear-button svelte-al0bnp" aria-label="Remove this file">×</button></td>`);
var root_1$1 = from_html(`<tr><td class="filename svelte-al0bnp"><!> <span class="stem svelte-al0bnp"> </span> <span class="ext svelte-al0bnp"> </span></td><td class="download svelte-al0bnp"><!></td><!></tr>`);
var root = from_html(`<div class="file-preview-holder svelte-al0bnp"><table class="file-preview svelte-al0bnp"><tbody class="svelte-al0bnp"></tbody></table></div>`);

function FilePreview($$anchor, $$props) {
	push($$props, true);

	let value = prop($$props, 'value', 7),
		selectable = prop($$props, 'selectable', 3, false),
		height = prop($$props, 'height', 3, undefined),
		allow_reordering = prop($$props, 'allow_reordering', 3, false);

	let dragging_index = state(null);
	let drop_target_index = state(null);

	function handle_drag_start(event, index) {
		set(dragging_index, index, true);

		if (event.dataTransfer) {
			event.dataTransfer.effectAllowed = "move";
			event.dataTransfer.setData("text/plain", index.toString());
		}
	}

	function handle_drag_over(event, index) {
		event.preventDefault();

		if (index === get(normalized_files).length - 1) {
			const rect = event.currentTarget.getBoundingClientRect();
			const midY = rect.top + rect.height / 2;

			set(drop_target_index, event.clientY > midY ? get(normalized_files).length : index, true);
		} else {
			set(drop_target_index, index, true);
		}

		if (event.dataTransfer) {
			event.dataTransfer.dropEffect = "move";
		}
	}

	function handle_drag_end(event) {
		if (!event.dataTransfer?.dropEffect || event.dataTransfer.dropEffect === "none") {
			set(dragging_index, null);
			set(drop_target_index, null);
		}
	}

	function handle_drop(event, index) {
		event.preventDefault();

		if (get(dragging_index) === null || get(dragging_index) === index) return;

		const files = Array.isArray(value()) ? [...value()] : [value()];
		const [removed] = files.splice(get(dragging_index), 1);

		files.splice(get(drop_target_index) === get(normalized_files).length ? get(normalized_files).length : index, 0, removed);

		const new_value = Array.isArray(value()) ? files : files[0];

		$$props.onchange?.(new_value);
		set(dragging_index, null);
		set(drop_target_index, null);
	}

	function split_filename(filename) {
		const last_dot = filename.lastIndexOf(".");

		if (last_dot === -1) {
			return [filename, ""];
		}

		return [filename.slice(0, last_dot), filename.slice(last_dot)];
	}

	let normalized_files = user_derived(() => (Array.isArray(value()) ? value() : [value()]).map((file) => {
		const [filename_stem, filename_ext] = split_filename(file.orig_name ?? "");

		return { ...file, filename_stem, filename_ext };
	}));

	function handle_row_click(event, index) {
		const tr = event.currentTarget;

		const should_select = event.target === tr || // Only select if the click is on the row itself
		tr && tr.firstElementChild && event.composedPath().includes(tr.firstElementChild); // Or if the click is on the name column

		if (should_select) {
			$$props.onselect?.({ value: get(normalized_files)[index].orig_name, index });
		}
	}

	function remove_file(index) {
		const files = Array.isArray(value()) ? [...value()] : [value()];
		const removed = files.splice(index, 1);
		const new_value = Array.isArray(value()) ? files : files[0];

		value(new_value);
		$$props.ondelete?.(removed[0]);
		$$props.onchange?.(new_value);
	}

	function handle_download(file) {
		$$props.ondownload?.(file);
	}

	const is_browser = typeof window !== "undefined";
	var div = root();
	let styles;
	var table = child(div);
	var tbody = child(table);

	each(tbody, 23, () => get(normalized_files), (file) => file.url, ($$anchor, file, i) => {
		var tr_1 = root_1$1();
		let classes;

		tr_1.__click = (event) => {
			handle_row_click(event, get(i));
		};

		var td = child(tr_1);
		var node = child(td);

		{
			var consequent = ($$anchor) => {
				var span = root_2();

				append($$anchor, span);
			};

			if_block(node, ($$render) => {
				if (allow_reordering() && get(normalized_files).length > 1) $$render(consequent);
			});
		}

		var span_1 = sibling(node, 2);
		var text$1 = child(span_1, true);

		reset(span_1);

		var span_2 = sibling(span_1, 2);
		var text_1 = child(span_2, true);

		reset(span_2);
		reset(td);

		var td_1 = sibling(td);
		var node_1 = child(td_1);

		{
			var consequent_1 = ($$anchor) => {
				{
					let $0 = user_derived(() => is_browser && window.__is_colab__ ? null : get(file).orig_name);

					DownloadLink($$anchor, {
						get href() {
							return get(file).url;
						},

						get download() {
							return get($0);
						},
						$$events: { click: () => handle_download(get(file)) },
						children: ($$anchor, $$slotProps) => {
							var fragment_1 = root_4$1();
							var node_2 = first_child(fragment_1);

							html(node_2, () => get(file).size != null ? prettyBytes(get(file).size) : "(size unknown)");
							next();
							append($$anchor, fragment_1);
						},
						$$slots: { default: true }
					});
				}
			};

			var alternate = ($$anchor) => {
				var text_2 = text();

				template_effect(($0) => set_text(text_2, $0), [() => $$props.i18n("file.uploading")]);
				append($$anchor, text_2);
			};

			if_block(node_1, ($$render) => {
				if (get(file).url) $$render(consequent_1); else $$render(alternate, false);
			});
		}

		reset(td_1);

		var node_3 = sibling(td_1);

		{
			var consequent_2 = ($$anchor) => {
				var td_2 = root_6();
				var button = child(td_2);

				button.__click = () => {
					remove_file(get(i));
				};

				button.__keydown = (event) => {
					if (event.key === "Enter") {
						remove_file(get(i));
					}
				};

				reset(td_2);
				append($$anchor, td_2);
			};

			if_block(node_3, ($$render) => {
				if (get(normalized_files).length > 1) $$render(consequent_2);
			});
		}

		reset(tr_1);

		template_effect(() => {
			classes = set_class(tr_1, 1, 'file svelte-al0bnp', null, classes, {
				selectable: selectable(),
				dragging: get(dragging_index) === get(i),
				'drop-target': get(drop_target_index) === get(i) || get(i) === get(normalized_files).length - 1 && get(drop_target_index) === get(normalized_files).length
			});

			set_attribute(tr_1, 'data-drop-target', get(drop_target_index) === get(normalized_files).length && get(i) === get(normalized_files).length - 1
				? "after"
				: get(drop_target_index) === get(i) + 1 ? "after" : "before");

			set_attribute(tr_1, 'draggable', allow_reordering() && get(normalized_files).length > 1);
			set_attribute(td, 'aria-label', get(file).orig_name);
			set_text(text$1, get(file).filename_stem);
			set_text(text_1, get(file).filename_ext);
		});

		event('dragstart', tr_1, (event) => handle_drag_start(event, get(i)));
		event('dragenter', tr_1, (event) => event.preventDefault());
		event('dragover', tr_1, (event) => handle_drag_over(event, get(i)));
		event('drop', tr_1, (event) => handle_drop(event, get(i)));
		event('dragend', tr_1, handle_drag_end);
		append($$anchor, tr_1);
	});

	reset(tbody);
	reset(table);
	reset(div);

	template_effect(() => styles = set_style(div, '', styles, {
		'max-height': height()
			? typeof height() === "number" ? height() + "px" : height()
			: "auto"
	}));

	append($$anchor, div);
	pop();
}

delegate(['click', 'keydown']);

var root_4 = from_html(`<!> <!>`, 1);
var root_3 = from_html(`<!> <!>`, 1);
var root_1 = from_html(`<!> <!> <!>`, 1);

function FileUpload($$anchor, $$props) {
	push($$props, true);

	let value = prop($$props, 'value', 15),
		show_label = prop($$props, 'show_label', 3, true),
		file_count = prop($$props, 'file_count', 3, "single"),
		file_types = prop($$props, 'file_types', 3, null),
		selectable = prop($$props, 'selectable', 3, false),
		height = prop($$props, 'height', 3, undefined),
		max_file_size = prop($$props, 'max_file_size', 3, null),
		uploading = prop($$props, 'uploading', 15, false),
		allow_reordering = prop($$props, 'allow_reordering', 3, false),
		upload_promise = prop($$props, 'upload_promise', 15),
		buttons = prop($$props, 'buttons', 3, null),
		on_custom_button_click = prop($$props, 'on_custom_button_click', 3, null);

	async function handle_upload(detail) {
		if (Array.isArray(value())) {
			value([...value(), ...Array.isArray(detail) ? detail : [detail]]);
		} else if (value()) {
			value([value(), ...Array.isArray(detail) ? detail : [detail]]);
		} else {
			value(detail);
		}

		await tick();
		$$props.onchange?.(value());
		$$props.onupload?.(value());
	}

	function handle_clear() {
		value(null);
		$$props.onchange?.(null);
		$$props.onclear?.();
	}

	let dragging = state(false);
	var fragment = root_1();
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
			if (show_label() && buttons() && buttons().length > 0) $$render(consequent);
		});
	}

	var node_1 = sibling(node, 2);

	{
		let $0 = user_derived(() => !value());
		let $1 = user_derived(() => $$props.label || "File");

		BlockLabel(node_1, {
			get show_label() {
				return show_label();
			},

			get Icon() {
				return File;
			},

			get float() {
				return get($0);
			},

			get label() {
				return get($1);
			}
		});
	}

	var node_2 = sibling(node_1, 2);

	{
		var consequent_2 = ($$anchor) => {
			var fragment_2 = root_3();
			var node_3 = first_child(fragment_2);

			{
				let $0 = user_derived(() => buttons() || []);

				IconButtonWrapper(node_3, {
					get buttons() {
						return get($0);
					},

					get on_custom_button_click() {
						return on_custom_button_click();
					},

					children: ($$anchor, $$slotProps) => {
						var fragment_3 = root_4();
						var node_4 = first_child(fragment_3);

						{
							var consequent_1 = ($$anchor) => {
								{
									let $0 = user_derived(() => $$props.i18n("common.upload"));

									IconButton($$anchor, {
										get Icon() {
											return Upload$1;
										},

										get label() {
											return get($0);
										},

										children: ($$anchor, $$slotProps) => {
											Upload($$anchor, {
												icon_upload: true,
												onload: handle_upload,
												get filetype() {
													return file_types();
												},

												get file_count() {
													return file_count();
												},

												get max_file_size() {
													return max_file_size();
												},

												get root() {
													return $$props.root;
												},

												get onerror() {
													return $$props.onerror;
												},

												get stream_handler() {
													return $$props.stream_handler;
												},

												get upload() {
													return $$props.upload;
												},

												get upload_promise() {
													return upload_promise();
												},

												set upload_promise($$value) {
													upload_promise($$value);
												},

												get dragging() {
													return get(dragging);
												},

												set dragging($$value) {
													set(dragging, $$value, true);
												},

												get uploading() {
													return uploading();
												},

												set uploading($$value) {
													uploading($$value);
												}
											});
										},
										$$slots: { default: true }
									});
								}
							};

							if_block(node_4, ($$render) => {
								if (!(file_count() === "single" && (Array.isArray(value()) ? value().length > 0 : value() !== null))) $$render(consequent_1);
							});
						}

						var node_5 = sibling(node_4, 2);

						{
							let $0 = user_derived(() => $$props.i18n("common.clear"));

							IconButton(node_5, {
								get Icon() {
									return Clear;
								},

								get label() {
									return get($0);
								},

								onclick: (event) => {
									event.stopPropagation();
									handle_clear();
								}
							});
						}

						append($$anchor, fragment_3);
					},
					$$slots: { default: true }
				});
			}

			var node_6 = sibling(node_3, 2);

			FilePreview(node_6, {
				get i18n() {
					return $$props.i18n;
				},

				get onselect() {
					return $$props.onselect;
				},

				get selectable() {
					return selectable();
				},

				get value() {
					return value();
				},

				get height() {
					return height();
				},

				get onchange() {
					return $$props.onchange;
				},

				get ondelete() {
					return $$props.ondelete;
				},

				get allow_reordering() {
					return allow_reordering();
				}
			});

			append($$anchor, fragment_2);
		};

		var alternate = ($$anchor) => {
			Upload($$anchor, {
				onload: handle_upload,
				get filetype() {
					return file_types();
				},

				get file_count() {
					return file_count();
				},

				get max_file_size() {
					return max_file_size();
				},

				get root() {
					return $$props.root;
				},

				get onerror() {
					return $$props.onerror;
				},

				get stream_handler() {
					return $$props.stream_handler;
				},

				get upload() {
					return $$props.upload;
				},

				get height() {
					return height();
				},

				get upload_promise() {
					return upload_promise();
				},

				set upload_promise($$value) {
					upload_promise($$value);
				},

				get dragging() {
					return get(dragging);
				},

				set dragging($$value) {
					set(dragging, $$value, true);
				},

				get uploading() {
					return uploading();
				},

				set uploading($$value) {
					uploading($$value);
				},

				children: ($$anchor, $$slotProps) => {
					var fragment_7 = comment();
					var node_7 = first_child(fragment_7);

					slot(node_7, $$props, 'default', {});
					append($$anchor, fragment_7);
				},
				$$slots: { default: true }
			});
		};

		if_block(node_2, ($$render) => {
			if (value() && (Array.isArray(value()) ? value().length > 0 : true)) $$render(consequent_2); else $$render(alternate, false);
		});
	}

	append($$anchor, fragment);
	pop();
}

export { FilePreview as F, FileUpload as a };
//# sourceMappingURL=FileUpload-9lL-iwhh.js.map
