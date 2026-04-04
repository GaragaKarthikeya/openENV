import { i as if_block, b as set_class, F as set_value, p as prop, a as set_attribute, f as set_style } from './i18n-CMo5lpzy.js';
import { R as push, u as state, v as proxy, aa as onMount, w as get, x as set, ae as onDestroy, y as user_effect, V as child, Y as reset, X as sibling, t as template_effect, a0 as set_text, a as append, T as pop, a5 as user_derived, W as from_html, a6 as comment, S as first_child, au as prepare_files, M as tick, aj as delegate } from './index-BvBk1Iap.js';
import { s as snippet } from './snippet-Bl5YVoCg.js';
import { a as action } from './actions-Cit_UA5S.js';

var root_2$1 = from_html(`<div class="file svelte-ua961l"><span><div class="progress-bar svelte-ua961l"><progress style="visibility:hidden;height:0;width:0;" max="100" class="svelte-ua961l"> </progress></div></span> <span class="file-name svelte-ua961l"> </span></div>`);
var root_1 = from_html(`<div><span class="uploading svelte-ua961l"> </span> <!></div>`);

function UploadProgress($$anchor, $$props) {
	push($$props, true);

	let stream;
	let progress = state(false);
	let current_file_upload = state(void 0);
	let file_to_display = user_derived(() => get(current_file_upload) || get(files_with_progress)[0]);

	let files_with_progress = state(proxy($$props.files.map((file) => {
		return { ...file, progress: 0 };
	})));

	function handleProgress(filename, chunk_size) {
		// Find the corresponding file in the array and update its progress
		set(
			files_with_progress,
			get(files_with_progress).map((file) => {
				if (file.orig_name === filename) {
					file.progress += chunk_size;
				}

				return file;
			}),
			true
		);
	}

	function getProgress(file) {
		return file.progress * 100 / (file.size || 0) || 0;
	}

	onMount(async () => {
		stream = await $$props.stream_handler(new URL(`${$$props.root}/gradio_api/upload_progress?upload_id=${$$props.upload_id}`));

		if (stream == null) {
			throw new Error("Event source is not defined");
		}

		// Event listener for progress updates
		stream.onmessage = async function (event) {
			const _data = JSON.parse(event.data);

			if (!get(progress)) set(progress, true);

			if (_data.msg === "done") {
				// the stream will close itself but is here for clarity; remove .close() in 5.0
				stream?.close();

				$$props.ondone?.();
			} else {
				set(current_file_upload, _data, true);
				handleProgress(_data.orig_name, _data.chunk_size);
			}
		};
	});

	onDestroy(() => {
		// the stream will close itself but is here for clarity; remove .close() in 5.0
		if (stream != null || stream != undefined) stream.close();
	});

	function calculateTotalProgress(files) {
		let totalProgress = 0;

		files.forEach((file) => {
			totalProgress += getProgress(file);
		});

		document.documentElement.style.setProperty("--upload-progress-width", (totalProgress / files.length).toFixed(2) + "%");

		return totalProgress / files.length;
	}

	user_effect(() => {
		calculateTotalProgress(get(files_with_progress));
	});

	var div = root_1();
	let classes;
	var span = child(div);
	var text = child(span);

	reset(span);

	var node = sibling(span, 2);

	{
		var consequent = ($$anchor) => {
			var div_1 = root_2$1();
			var span_1 = child(div_1);
			var div_2 = child(span_1);
			var progress_1 = child(div_2);
			var text_1 = child(progress_1, true);

			reset(progress_1);
			reset(div_2);
			reset(span_1);

			var span_2 = sibling(span_1, 2);
			var text_2 = child(span_2, true);

			reset(span_2);
			reset(div_1);

			template_effect(
				($0, $1) => {
					set_value(progress_1, $0);
					set_text(text_1, $1);
					set_text(text_2, get(file_to_display).orig_name);
				},
				[
					() => getProgress(get(file_to_display)),
					() => getProgress(get(file_to_display))
				]
			);

			append($$anchor, div_1);
		};

		if_block(node, ($$render) => {
			if (get(file_to_display)) $$render(consequent);
		});
	}

	reset(div);

	template_effect(() => {
		classes = set_class(div, 1, 'wrap svelte-ua961l', null, classes, { progress: get(progress) });

		set_text(text, `Uploading ${get(files_with_progress).length ?? ''}
		${get(files_with_progress).length > 1 ? "files" : "file"}...`);
	});

	append($$anchor, div);
	pop();
}

function create_drag() {
  let hidden_input;
  let _options;
  return {
    drag(node, options = {}) {
      _options = options;
      function setup_hidden_input() {
        hidden_input = document.createElement("input");
        hidden_input.type = "file";
        hidden_input.style.display = "none";
        hidden_input.setAttribute("aria-label", "File upload");
        hidden_input.setAttribute("data-testid", "file-upload");
        const accept_options = Array.isArray(_options.accepted_types) ? _options.accepted_types.join(",") : _options.accepted_types || void 0;
        if (accept_options) {
          hidden_input.accept = accept_options;
        }
        hidden_input.multiple = _options.mode === "multiple" || false;
        if (_options.mode === "directory") {
          hidden_input.webkitdirectory = true;
          hidden_input.setAttribute("directory", "");
          hidden_input.setAttribute("mozdirectory", "");
        }
        node.appendChild(hidden_input);
      }
      setup_hidden_input();
      function handle_drag(e) {
        e.preventDefault();
        e.stopPropagation();
      }
      function handle_drag_enter(e) {
        e.preventDefault();
        e.stopPropagation();
        _options.on_drag_change?.(true);
      }
      function handle_drag_leave(e) {
        e.preventDefault();
        e.stopPropagation();
        _options.on_drag_change?.(false);
      }
      function handle_drop(e) {
        e.preventDefault();
        e.stopPropagation();
        _options.on_drag_change?.(false);
        if (!e.dataTransfer?.files) return;
        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) {
          _options.on_files?.(files);
        }
      }
      function handle_click() {
        if (!_options.disable_click) {
          hidden_input.value = "";
          hidden_input.click();
        }
      }
      function handle_file_input_change() {
        if (hidden_input.files) {
          const files = Array.from(hidden_input.files);
          if (files.length > 0) {
            _options.on_files?.(files);
          }
        }
      }
      node.addEventListener("drag", handle_drag);
      node.addEventListener("dragstart", handle_drag);
      node.addEventListener("dragend", handle_drag);
      node.addEventListener("dragover", handle_drag);
      node.addEventListener("dragenter", handle_drag_enter);
      node.addEventListener("dragleave", handle_drag_leave);
      node.addEventListener("drop", handle_drop);
      node.addEventListener("click", handle_click);
      hidden_input.addEventListener("change", handle_file_input_change);
      return {
        update(new_options) {
          _options = new_options;
          hidden_input.remove();
          setup_hidden_input();
          hidden_input.addEventListener("change", handle_file_input_change);
        },
        destroy() {
          node.removeEventListener("drag", handle_drag);
          node.removeEventListener("dragstart", handle_drag);
          node.removeEventListener("dragend", handle_drag);
          node.removeEventListener("dragover", handle_drag);
          node.removeEventListener("dragenter", handle_drag_enter);
          node.removeEventListener("dragleave", handle_drag_leave);
          node.removeEventListener("drop", handle_drop);
          node.removeEventListener("click", handle_click);
          hidden_input.removeEventListener("change", handle_file_input_change);
          hidden_input.remove();
        }
      };
    },
    open_file_upload() {
      if (hidden_input) {
        hidden_input.value = "";
        hidden_input.click();
      }
    }
  };
}

var root_2 = from_html(`<button><!></button>`);
var root_7 = from_html(`<button aria-dropeffect="copy"><!></button>`);

function Upload($$anchor, $$props) {
	push($$props, true);

	const { drag, open_file_upload: _open_file_upload } = create_drag();

	let filetype = prop($$props, 'filetype', 3, null),
		dragging = prop($$props, 'dragging', 15, false),
		boundedheight = prop($$props, 'boundedheight', 3, true),
		center = prop($$props, 'center', 3, true),
		flex = prop($$props, 'flex', 3, true),
		file_count = prop($$props, 'file_count', 3, "single"),
		disable_click = prop($$props, 'disable_click', 3, false),
		hidden = prop($$props, 'hidden', 3, false),
		format = prop($$props, 'format', 3, "file"),
		uploading = prop($$props, 'uploading', 15, false),
		show_progress = prop($$props, 'show_progress', 3, true),
		max_file_size = prop($$props, 'max_file_size', 3, null),
		icon_upload = prop($$props, 'icon_upload', 3, false),
		height = prop($$props, 'height', 3, undefined),
		aria_label = prop($$props, 'aria_label', 3, undefined),
		upload_promise = prop($$props, 'upload_promise', 15);

	function open_upload() {
		_open_file_upload();
	}

	let upload_id = "";
	let file_data;
	let accept_file_types = state(null);
	let use_post_upload_validation = null;

	const get_ios = () => {
		if (typeof navigator !== "undefined") {
			const userAgent = navigator.userAgent.toLowerCase();

			return userAgent.indexOf("iphone") > -1 || userAgent.indexOf("ipad") > -1;
		}

		return false;
	};

	let ios = get_ios();
	const validFileTypes = ["image", "video", "audio", "text", "file"];

	const process_file_type = (type) => {
		if (ios && type.startsWith(".")) {
			use_post_upload_validation = true;

			return type;
		}

		if (ios && type.includes("file/*")) {
			return "*";
		}

		if (type.startsWith(".") || type.endsWith("/*")) {
			return type;
		}

		if (validFileTypes.includes(type)) {
			return type + "/*";
		}

		return "." + type;
	};

	user_effect(() => {
		if (filetype() == null) {
			set(accept_file_types, null);
		} else if (typeof filetype() === "string") {
			set(accept_file_types, process_file_type(filetype()), true);
		} else if (ios && filetype().includes("file/*")) {
			set(accept_file_types, "*");
		} else {
			const processed = filetype().map(process_file_type);

			set(accept_file_types, processed.join(", "), true);
		}
	});

	function paste_clipboard() {
		navigator.clipboard.read().then(async (items) => {
			for (let i = 0; i < items.length; i++) {
				const type = items[i].types.find((t) => t.startsWith("image/"));

				if (type) {
					items[i].getType(type).then(async (blob) => {
						const file = new File([blob], `clipboard.${type.replace("image/", "")}`);

						await load_files([file]);
					});

					break;
				}
			}
		});
	}

	function open_file_upload() {
		_open_file_upload();
	}

	async function handle_upload(file_data, _upload_id) {
		if (!_upload_id) {
			upload_id = Math.random().toString(36).substring(2, 15);
		} else {
			upload_id = _upload_id;
		}

		await tick();
		uploading(true);

		upload_promise(new Promise(async (resolve) => {
			try {
				const _file_data = await $$props.upload(file_data, $$props.root, upload_id, max_file_size() ?? Infinity);

				$$props.onload?.(file_count() === "single" ? _file_data?.[0] : _file_data);
				resolve(_file_data || []);
				uploading(false);
			} catch(e) {
				$$props.onerror?.(e.message);
				uploading(false);
				resolve([]);
			}
		}));

		return upload_promise();
	}

	function is_valid_mimetype(file_accept, uploaded_file_extension, uploaded_file_type) {
		if (!file_accept || file_accept === "*" || file_accept === "file/*" || Array.isArray(file_accept) && file_accept.some((accept) => accept === "*" || accept === "file/*")) {
			return true;
		}

		let acceptArray;

		if (typeof file_accept === "string") {
			acceptArray = file_accept.split(",").map((s) => s.trim());
		} else if (Array.isArray(file_accept)) {
			acceptArray = file_accept;
		} else {
			return false;
		}

		return acceptArray.includes(uploaded_file_extension) || acceptArray.some((type) => {
			const [category] = type.split("/").map((s) => s.trim());

			return type.endsWith("/*") && uploaded_file_type.startsWith(category + "/");
		});
	}

	async function load_files(files, upload_id) {
		if (!files.length) {
			return;
		}

		let _files = files.map((f) => new File([f], f instanceof File ? f.name : "file", { type: f.type }));

		if (ios && use_post_upload_validation) {
			_files = _files.filter((file) => {
				if (is_valid_file(file)) {
					return true;
				}

				$$props.onerror?.(`Invalid file type: ${file.name}. Only ${filetype()} allowed.`);

				return false;
			});

			if (_files.length === 0) {
				return [];
			}
		}

		file_data = await prepare_files(_files);

		return await handle_upload(file_data, upload_id);
	}

	function is_valid_file(file) {
		if (!filetype()) return true;

		const allowed_types = Array.isArray(filetype()) ? filetype() : [filetype()];

		return allowed_types.some((type) => {
			const processed_type = process_file_type(type);

			if (processed_type.startsWith(".")) {
				return file.name.toLowerCase().endsWith(processed_type.toLowerCase());
			}

			if (processed_type === "*") {
				return true;
			}

			if (processed_type.endsWith("/*")) {
				const [category] = processed_type.split("/");

				return file.type.startsWith(category + "/");
			}

			return file.type === processed_type;
		});
	}

	async function load_files_from_upload(files) {
		const files_to_load = files.filter((file) => {
			const file_extension = "." + file.name.toLowerCase().split(".").pop();

			if (file_extension && is_valid_mimetype(get(accept_file_types), file_extension, file.type)) {
				return true;
			}

			if (file_extension && Array.isArray(filetype())
				? filetype().includes(file_extension)
				: file_extension === filetype()) {
				return true;
			}

			$$props.onerror?.(`Invalid file type only ${filetype()} allowed.`);

			return false;
		});

		if (format() != "blob") {
			await load_files(files_to_load);
		} else {
			if (file_count() === "single") {
				$$props.onload?.(files_to_load[0]);

				return;
			}

			$$props.onload?.(files_to_load);
		}
	}

	async function load_files_from_drop(e) {
		dragging(false);

		if (!e.dataTransfer?.files) return;

		const files_to_load = Array.from(e.dataTransfer.files).filter(is_valid_file);

		if (format() != "blob") {
			await load_files(files_to_load);
		} else {
			if (file_count() === "single") {
				$$props.onload?.(files_to_load[0]);

				return;
			}

			$$props.onload?.(files_to_load);
		}
	}

	var $$exports = {
		open_upload,
		paste_clipboard,
		open_file_upload,
		load_files,
		load_files_from_drop
	};

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent_1 = ($$anchor) => {
			var button = root_2();

			button.__click = paste_clipboard;

			let classes;
			let styles;
			var node_1 = child(button);

			{
				var consequent = ($$anchor) => {
					var fragment_1 = comment();
					var node_2 = first_child(fragment_1);

					snippet(node_2, () => $$props.children);
					append($$anchor, fragment_1);
				};

				if_block(node_1, ($$render) => {
					if ($$props.children) $$render(consequent);
				});
			}

			reset(button);

			template_effect(() => {
				set_attribute(button, 'tabindex', hidden() ? -1 : 0);
				set_attribute(button, 'aria-label', aria_label() || "Paste from clipboard");

				classes = set_class(button, 1, 'svelte-8prmba', null, classes, {
					hidden: hidden(),
					center: center(),
					boundedheight: boundedheight(),
					flex: flex(),
					'icon-mode': icon_upload()
				});

				styles = set_style(button, '', styles, {
					height: icon_upload()
						? ""
						: height()
							? typeof height() === "number" ? height() + "px" : height()
							: "100%"
				});
			});

			append($$anchor, button);
		};

		var alternate_1 = ($$anchor) => {
			var fragment_2 = comment();
			var node_3 = first_child(fragment_2);

			{
				var consequent_3 = ($$anchor) => {
					var fragment_3 = comment();
					var node_4 = first_child(fragment_3);

					{
						var consequent_2 = ($$anchor) => {
							UploadProgress($$anchor, {
								get root() {
									return $$props.root;
								},

								get upload_id() {
									return upload_id;
								},

								get files() {
									return file_data;
								},

								get stream_handler() {
									return $$props.stream_handler;
								}
							});
						};

						if_block(node_4, ($$render) => {
							if (!hidden()) $$render(consequent_2);
						});
					}

					append($$anchor, fragment_3);
				};

				var alternate = ($$anchor) => {
					var button_1 = root_7();
					let classes_1;
					let styles_1;
					var node_5 = child(button_1);

					{
						var consequent_4 = ($$anchor) => {
							var fragment_5 = comment();
							var node_6 = first_child(fragment_5);

							snippet(node_6, () => $$props.children);
							append($$anchor, fragment_5);
						};

						if_block(node_5, ($$render) => {
							if ($$props.children) $$render(consequent_4);
						});
					}

					reset(button_1);

					action(button_1, ($$node, $$action_arg) => drag?.($$node, $$action_arg), () => ({
						on_drag_change: (d) => dragging(d),
						on_files: (files) => load_files_from_upload(files),
						accepted_types: get(accept_file_types),
						mode: file_count(),
						disable_click: disable_click()
					}));

					template_effect(() => {
						set_attribute(button_1, 'tabindex', hidden() ? -1 : 0);
						set_attribute(button_1, 'aria-label', aria_label() || "Click to upload or drop files");

						classes_1 = set_class(button_1, 1, 'svelte-8prmba', null, classes_1, {
							hidden: hidden(),
							center: center(),
							boundedheight: boundedheight(),
							flex: flex(),
							disable_click: disable_click(),
							'icon-mode': icon_upload()
						});

						styles_1 = set_style(button_1, '', styles_1, {
							height: icon_upload()
								? ""
								: height()
									? typeof height() === "number" ? height() + "px" : height()
									: "100%"
						});
					});

					append($$anchor, button_1);
				};

				if_block(
					node_3,
					($$render) => {
						if (uploading() && show_progress()) $$render(consequent_3); else $$render(alternate, false);
					},
					true
				);
			}

			append($$anchor, fragment_2);
		};

		if_block(node, ($$render) => {
			if (filetype() === "clipboard") $$render(consequent_1); else $$render(alternate_1, false);
		});
	}

	append($$anchor, fragment);

	return pop($$exports);
}

delegate(['click']);

export { Upload as U, UploadProgress as a, create_drag as c };
//# sourceMappingURL=Upload-CPeZiqam.js.map
