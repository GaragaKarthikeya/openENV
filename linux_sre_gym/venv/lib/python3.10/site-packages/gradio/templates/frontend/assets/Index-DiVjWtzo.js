import { p as prop, i as if_block, a as set_attribute, d as bind_this, k as each, u as index, N as bind_select_value, b as set_class, f as set_style, r as rest_props, g as spread_props } from './i18n-CMo5lpzy.js';
import { a as append, f as from_svg, M as tick, aj as delegate, R as push, u as state, v as proxy, aa as onMount, x as set, w as get, ae as onDestroy, y as user_effect, V as child, Y as reset, t as template_effect, T as pop, X as sibling, S as first_child, a0 as set_text, au as prepare_files, W as from_html, aN as update, ai as user_pre_effect, a7 as text, A as effect, Z as event, a5 as user_derived, a8 as next, a6 as comment, ak as remove_textarea_child } from './index-BvBk1Iap.js';
import { f as format_time, s as should_show_scroll_fade, G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { a as action } from './actions-Cit_UA5S.js';
import { b as bind_value } from './input-C317uaXR.js';
import { M as Microphone, V as Video } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { B as BlockTitle } from './BlockTitle-7XQ-8ENg.js';
import { A as ArrowUp } from './ArrowUp-D6YcMimJ.js';
import { C as Check } from './Check-BlnBgjLT.js';
import { C as Clear } from './Clear-Dgs5UhTO.js';
import { F as File$1 } from './File-_cC4Z_uD.js';
import { M as Music } from './Music-D0NN2hV9.js';
import { S as Square } from './Square-B6INQV79.js';
import { S as ScrollFade } from './ScrollFade-BJkuhu7T.js';
import { U as Upload } from './Upload-CPeZiqam.js';
import { I as Image } from './Image-CDw5qb5O.js';
/* empty css                                                    */
import { u, p as process_audio, M as MinimalAudioPlayer } from './MinimalAudioPlayer-Cdqg8MIC.js';
import { o } from './record.esm-BVY9RaAH.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
export { default as BaseExample } from './Example-BqGmQLY6.js';
import './clone-DaB-S2nH.js';
import './snippet-Bl5YVoCg.js';
import './Info-BrkJhX6_.js';
import './html-CdUEtR5E.js';
import './misc-ByVveAau.js';
/* empty css                                             */
import './index-C7e-J7CF.js';
import './size-Cu4PabCL.js';
import './Video-B3xDM6GN.js';

var root = from_svg(`<svg fill="currentColor" width="100%" height="100%" viewBox="0 0 1920 1920" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"><path d="M1752.768 221.109C1532.646.986 1174.283.986 954.161 221.109l-838.588 838.588c-154.052 154.165-154.052 404.894 0 558.946 149.534 149.421 409.976 149.308 559.059 0l758.738-758.626c87.982-88.094 87.982-231.417 0-319.51-88.32-88.208-231.642-87.982-319.51 0l-638.796 638.908 79.85 79.849 638.795-638.908c43.934-43.821 115.539-43.934 159.812 0 43.934 44.047 43.934 115.877 0 159.812l-758.739 758.625c-110.23 110.118-289.355 110.005-399.36 0-110.118-110.117-110.005-289.242 0-399.247l838.588-838.588c175.963-175.962 462.382-176.188 638.909 0 176.075 176.188 176.075 462.833 0 638.908l-798.607 798.72 79.849 79.85 798.607-798.72c220.01-220.123 220.01-578.485 0-798.607" fill-rule="evenodd"></path></g></svg>`);

function Paperclip($$anchor) {
	var svg = root();

	append($$anchor, svg);
}

async function resize(target, lines, max_lines) {
  await tick();
  if (lines === max_lines) return;
  const computed_styles = window.getComputedStyle(target);
  const padding_top = parseFloat(computed_styles.paddingTop);
  const padding_bottom = parseFloat(computed_styles.paddingBottom);
  const line_height = parseFloat(computed_styles.lineHeight);
  let max = max_lines === void 0 ? false : padding_top + padding_bottom + line_height * max_lines;
  let min = padding_top + padding_bottom + lines * line_height;
  target.style.height = "1px";
  let scroll_height;
  if (max && target.scrollHeight > max) {
    scroll_height = max;
  } else if (target.scrollHeight < min) {
    scroll_height = min;
  } else {
    scroll_height = target.scrollHeight;
  }
  target.style.height = `${scroll_height}px`;
}
function text_area_resize(_el, _value) {
  if (_value.lines === _value.max_lines) return;
  _el.style.overflowY = "scroll";
  function handle_input(event) {
    resize(event.target, _value.lines, _value.max_lines);
  }
  _el.addEventListener("input", handle_input);
  if (!_value.text.trim()) return;
  resize(_el, _value.lines, _value.max_lines);
  return {
    destroy: () => _el.removeEventListener("input", handle_input)
  };
}

var root_4$1 = from_html(`<option> </option>`);
var root_3$1 = from_html(`<select class="device-select-large svelte-doy9oc" aria-label="Select input device"></select>`);
var root_2 = from_html(`<div class="device-selection-wrapper svelte-doy9oc"><!> <button class="record-button svelte-doy9oc" aria-label="Start recording"></button></div>`);
var root_5$1 = from_html(`<div class="waveform-wrapper svelte-doy9oc"></div> <div class="timestamp svelte-doy9oc"> </div> <button class="stop-button svelte-doy9oc" aria-label="Stop recording"><!></button>`, 1);
var root_1$2 = from_html(`<div class="minimal-audio-recorder svelte-doy9oc" data-testid="minimal-audio-recorder"><!></div>`);

function MinimalAudioRecorder($$anchor, $$props) {
	push($$props, true);

	let waveform_settings = prop($$props, 'waveform_settings', 19, () => ({})),
		recording = prop($$props, 'recording', 15),
		max_file_size = prop($$props, 'max_file_size', 3, null),
		upload_promise = prop($$props, 'upload_promise', 15);

	let container;
	let waveform;
	let record = state(void 0);
	let seconds = state(0);
	let interval;
	let is_recording = state(false);
	let has_started = state(false);
	let mic_devices = state(proxy([]));
	let selected_device_id = state("");
	let show_device_selection = state(false);

	const start_interval = () => {
		clearInterval(interval);

		interval = setInterval(
			() => {
				update(seconds);
			},
			1000
		);
	};

	const create_waveform = async () => {
		if (!container) return;

		if (waveform) {
			waveform.destroy();
		}

		const accentColor = getComputedStyle(document.documentElement).getPropertyValue("--color-accent") || "#ff7c00";

		waveform = u.create({
			container,
			height: 32,
			waveColor: "rgba(128, 128, 128, 0.5)",
			progressColor: accentColor,
			cursorColor: "transparent",
			barWidth: 2,
			barGap: 2,
			barRadius: 2,
			interact: false,
			hideScrollbar: true,
			...waveform_settings()
		});

		set(
			record,
			waveform.registerPlugin(o.create({
				scrollingWaveform: true,
				scrollingWaveformWindow: 7,
				renderRecordedAudio: false
			})),
			true
		);

		get(record).on("record-start", () => {
			start_interval();
			set(is_recording, true);
			set(has_started, true);
		});

		get(record).on("record-end", async (blob) => {
			clearInterval(interval);
			set(is_recording, false);

			upload_promise((async () => {
				try {
					const array_buffer = await blob.arrayBuffer();
					const context = new AudioContext({ sampleRate: waveform_settings().sampleRate || 44100 });
					const audio_buffer = await context.decodeAudioData(array_buffer);

					if (audio_buffer) {
						const audio = await process_audio(audio_buffer);
						const audio_blob = new File([audio], "audio.wav", { type: "audio/wav" });
						const prepared_files = await prepare_files([audio_blob], false);
						const uploaded_files = await $$props.upload(prepared_files, $$props.root, undefined, max_file_size() || undefined);
						const file_data = uploaded_files?.[0];

						if (file_data) {
							$$props.onchange?.(file_data);
						}
					}
				} catch(e) {
					console.error("Error processing audio:", e);
				} finally {
					$$props.onstoprecording?.();
					upload_promise(null);
				}
			})());

			await upload_promise();
		});
	};

	onMount(async () => {
		try {
			const devices = await o.getAvailableAudioDevices();

			set(mic_devices, devices.filter((device) => device.deviceId), true);

			if (get(mic_devices).length > 0) {
				set(selected_device_id, get(mic_devices)[0].deviceId, true);
			}

			if (get(mic_devices).length > 1) {
				set(show_device_selection, true);
			} else {
				await create_waveform();
			}
		} catch(err) {
			await create_waveform();
		}
	});

	onDestroy(() => {
		clearInterval(interval);

		if (get(record)) {
			get(record).stopMic();
		}

		if (waveform) {
			waveform.destroy();
		}
	});

	user_effect(() => {
		if (recording() && !get(is_recording) && get(record) && get(has_started) === false && get(mic_devices).length <= 1) {
			get(record).startMic({ deviceId: get(selected_device_id) }).then(() => {
				get(record)?.startRecording();
			}).catch((err) => {
				console.error("Failed to access microphone:", err);
				$$props.onclear?.();
			});
		} else if (!recording() && get(is_recording) && get(record)) {
			get(record).stopRecording();
			set(seconds, 0);
		}
	});

	async function startRecording() {
		set(show_device_selection, false);
		set(has_started, true);
		await tick();
		await create_waveform();

		if (!get(record)) return;

		try {
			await get(record).startMic({ deviceId: get(selected_device_id) });
			get(record).startRecording();
		} catch(err) {
			console.error("Error starting recording:", err);
			set(show_device_selection, get(mic_devices).length > 1);
			set(has_started, false);
			recording(false);
		}
	}

	var div = root_1$2();
	var node = child(div);

	{
		var consequent_1 = ($$anchor) => {
			var div_1 = root_2();
			var node_1 = child(div_1);

			{
				var consequent = ($$anchor) => {
					var select = root_3$1();

					each(select, 21, () => get(mic_devices), index, ($$anchor, device) => {
						var option = root_4$1();
						var text = child(option, true);

						reset(option);

						var option_value = {};

						template_effect(() => {
							set_text(text, get(device).label);

							if (option_value !== (option_value = get(device).deviceId)) {
								option.value = (option.__value = get(device).deviceId) ?? '';
							}
						});

						append($$anchor, option);
					});

					reset(select);
					bind_select_value(select, () => get(selected_device_id), ($$value) => set(selected_device_id, $$value));
					append($$anchor, select);
				};

				if_block(node_1, ($$render) => {
					if (get(mic_devices).length > 1) $$render(consequent);
				});
			}

			var button = sibling(node_1, 2);

			button.__click = startRecording;
			reset(div_1);
			append($$anchor, div_1);
		};

		var alternate = ($$anchor) => {
			var fragment = root_5$1();
			var div_2 = first_child(fragment);

			bind_this(div_2, ($$value) => container = $$value, () => container);

			var div_3 = sibling(div_2, 2);
			var text_1 = child(div_3, true);

			reset(div_3);

			var button_1 = sibling(div_3, 2);

			button_1.__click = () => {
				if (get(is_recording)) {
					recording(false);
				} else {
					$$props.onclear?.();
				}
			};

			var node_2 = child(button_1);

			Square(node_2, {});
			reset(button_1);
			template_effect(($0) => set_text(text_1, $0), [() => format_time(get(seconds))]);
			append($$anchor, fragment);
		};

		if_block(node, ($$render) => {
			if (get(show_device_selection)) $$render(consequent_1); else $$render(alternate, false);
		});
	}

	reset(div);
	template_effect(() => set_attribute(div, 'aria-label', $$props.label || "Audio Recorder"));
	append($$anchor, div);
	pop();
}

delegate(['click']);

var root_4 = from_html(`<div class="recording-content svelte-1qn0337"><!></div>`);
var root_5 = from_html(`<div class="recording-content svelte-1qn0337"><!> <div class="action-buttons svelte-1qn0337"><button class="confirm-button svelte-1qn0337" aria-label="Attach audio"><!></button> <button class="cancel-button svelte-1qn0337" aria-label="Clear audio"><!></button></div></div>`);
var root_3 = from_html(`<div><!></div>`);
var root_8 = from_html(`<button data-testid="upload-button" class="upload-button thumbnail-add svelte-1qn0337" aria-label="Upload a file"><!></button>`);
var root_9 = from_html(`<span class="thumbnail-wrapper svelte-1qn0337" role="listitem" aria-label="File thumbnail"><div class="thumbnail-item thumbnail-small svelte-1qn0337"><!> <button class="delete-button svelte-1qn0337" aria-label="Remove file"><!></button></div></span>`);
var root_16 = from_html(`<div class="loader svelte-1qn0337" role="status" aria-label="Uploading"></div>`);
var root_7 = from_html(`<div class="thumbnails svelte-1qn0337" aria-label="Uploaded files" data-testid="container_el"><!> <!> <!></div>`);
var root_17 = from_html(`<button data-testid="upload-button" class="upload-button icon-button svelte-1qn0337" aria-label="Upload a file"><!></button>`);
var root_18 = from_html(`<button data-testid="microphone-button" aria-label="Record audio"><!></button>`);
var root_19 = from_html(`<button data-testid="submit-button" aria-label="Submit"><!></button>`);
var root_22 = from_html(`<button aria-label="Stop"><!></button>`);
var root_1$1 = from_html(`<div role="group" aria-label="Multimedia input field"><!> <div class="input-container svelte-1qn0337"><!> <!> <div><!> <div class="input-row svelte-1qn0337"><!> <div class="textarea-wrapper svelte-1qn0337"><textarea data-testid="textbox"></textarea> <!></div> <!> <!> <!></div></div></div></div>`);

function MultimodalTextbox($$anchor, $$props) {
	push($$props, true);

	let value = prop($$props, 'value', 15),
		value_is_output = prop($$props, 'value_is_output', 3, false),
		lines = prop($$props, 'lines', 3, 1),
		placeholder = prop($$props, 'placeholder', 3, ""),
		disabled = prop($$props, 'disabled', 3, false),
		info = prop($$props, 'info', 3, undefined),
		show_label = prop($$props, 'show_label', 3, true),
		submit_btn = prop($$props, 'submit_btn', 3, null),
		stop_btn = prop($$props, 'stop_btn', 3, null),
		rtl = prop($$props, 'rtl', 3, false),
		autofocus = prop($$props, 'autofocus', 3, false),
		text_align = prop($$props, 'text_align', 3, undefined),
		autoscroll = prop($$props, 'autoscroll', 3, true),
		file_types_string = prop($$props, 'file_types_string', 3, null),
		max_file_size = prop($$props, 'max_file_size', 3, null),
		file_count = prop($$props, 'file_count', 3, "multiple"),
		max_plain_text_length = prop($$props, 'max_plain_text_length', 3, 1000);
		prop($$props, 'waveform_options', 19, () => ({ show_recording_waveform: true }));
		let sources_string = prop($$props, 'sources_string', 3, "upload"),
		active_source = prop($$props, 'active_source', 15),
		html_attributes = prop($$props, 'html_attributes', 3, null),
		upload_promise = prop($$props, 'upload_promise', 15),
		dragging = prop($$props, 'dragging', 15);

	let upload_component;
	let el;
	let can_scroll = state(false);
	let previous_scroll_top = state(0);
	let user_has_scrolled_up = state(false);
	let show_fade = state(false);
	let uploading = state(false);
	let oldValue = state(proxy(value()?.text ?? ""));
	let recording = state(false);
	let mic_audio = state(null);
	let full_container;
	let sources = user_derived(() => sources_string().split(",").map((s) => s.trim()).filter((s) => s === "upload" || s === "microphone"));

	let file_types = user_derived(() => file_types_string()
		? file_types_string().split(",").map((s) => s.trim())
		: null);

	let show_upload = user_derived(() => get(sources) && get(sources).includes("upload") && !(file_count() === "single" && value()?.files?.length > 0));

	function update_fade() {
		set(show_fade, should_show_scroll_fade(el), true);
	}

	user_effect(() => {
		if (el && value()?.text !== undefined) {
			tick().then(update_fade);
		}
	});

	user_effect(() => {
		$$props.ondrag?.(dragging());
	});

	user_effect(() => {
		if (value() && get(oldValue) !== value().text) {
			$$props.onchange?.(value());
			set(oldValue, value().text, true);
		}
	});

	user_effect(() => {
		if (el && lines() !== $$props.max_lines) {
			resize(el, lines(), $$props.max_lines);
		}
	});

	user_pre_effect(() => {
		if (el && el.offsetHeight + el.scrollTop > el.scrollHeight - 100) {
			set(can_scroll, true);
		}
	});

	const scroll = () => {
		if (get(can_scroll) && autoscroll() && !get(user_has_scrolled_up)) {
			el.scrollTo(0, el.scrollHeight);
		}
	};

	async function handle_change() {
		$$props.onchange?.(value());

		if (!value_is_output()) {
			$$props.oninput?.();
		}
	}

	user_effect(() => {
		value();

		if (autofocus() && el) {
			el.focus();
		}
	});

	user_effect(() => {
		if (get(can_scroll) && autoscroll()) {
			scroll();
		}
	});

	function handle_select(event) {
		const target = event.target;
		const text = target.value;
		const index = [target.selectionStart, target.selectionEnd];

		$$props.onselect?.({ value: text.substring(...index), index });
	}

	async function handle_keypress(e) {
		if (e.key === "Enter" && e.shiftKey && lines() > 1) {
			e.preventDefault();
			await tick();
			$$props.onsubmit?.();
		} else if (e.key === "Enter" && !e.shiftKey && lines() === 1 && $$props.max_lines >= 1) {
			e.preventDefault();
			add_mic_audio_to_files();
			active_source(null);
			await tick();
			$$props.onsubmit?.();
		}
	}

	function handle_scroll(event) {
		const target = event.target;
		const current_scroll_top = target.scrollTop;

		if (current_scroll_top < get(previous_scroll_top)) {
			set(user_has_scrolled_up, true);
		}

		set(previous_scroll_top, current_scroll_top, true);

		const max_scroll_top = target.scrollHeight - target.clientHeight;
		const user_has_scrolled_to_bottom = current_scroll_top >= max_scroll_top;

		if (user_has_scrolled_to_bottom) {
			set(user_has_scrolled_up, false);
		}

		update_fade();
	}

	async function handle_upload(detail) {
		handle_change();

		if (Array.isArray(detail)) {
			for (let file of detail) {
				value().files.push(file);
			}

			value(value());
		} else {
			value().files.push(detail);
			value(value());
		}

		await tick();
		$$props.onchange?.(value());
		$$props.onupload?.(detail);
	}

	function remove_thumbnail(event, index) {
		handle_change();
		event.stopPropagation();
		value().files.splice(index, 1);
		value(value());
	}

	function handle_upload_click() {
		upload_component.open_upload();
	}

	function handle_stop() {
		$$props.onstop?.();
	}

	function add_mic_audio_to_files() {
		if (get(mic_audio)) {
			value().files.push(get(mic_audio));
			value(value());
			set(mic_audio, null);
			$$props.onchange?.(value());
		}
	}

	function handle_submit() {
		add_mic_audio_to_files();
		active_source(null);
		$$props.onsubmit?.();
	}

	async function handle_paste(event) {
		if (!event.clipboardData) return;

		const items = event.clipboardData.items;
		const text = event.clipboardData.getData("text");

		if (text && text.length > max_plain_text_length()) {
			event.preventDefault();

			const file = new window.File([text], "pasted_text.txt", { type: "text/plain", lastModified: Date.now() });

			if (upload_component) {
				upload_component.load_files([file]);
			}

			return;
		}

		for (let index in items) {
			const item = items[index];

			if (item.kind === "file" && item.type.includes("image")) {
				const blob = item.getAsFile();

				if (blob) upload_component.load_files([blob]);
			}
		}
	}

	function handle_dragenter(event) {
		event.preventDefault();
		dragging(true);
	}

	function handle_dragleave(event) {
		event.preventDefault();

		const rect = full_container.getBoundingClientRect();
		const { clientX, clientY } = event;

		if (clientX <= rect.left || clientX >= rect.right || clientY <= rect.top || clientY >= rect.bottom) {
			dragging(false);
		}
	}

	function handle_drop(event) {
		event.preventDefault();
		dragging(false);

		if (event.dataTransfer && event.dataTransfer.files) {
			const files = Array.from(event.dataTransfer.files);

			if (get(file_types)) {
				const valid_files = files.filter((file) => {
					return get(file_types).some((type) => {
						if (type.startsWith(".")) {
							return file.name.toLowerCase().endsWith(type.toLowerCase());
						}

						return file.type.match(new RegExp(type.replace("*", ".*")));
					});
				});

				const invalid_files = files.length - valid_files.length;

				if (invalid_files > 0) {
					$$props.onerror?.(`${invalid_files} file(s) were rejected. Accepted formats: ${get(file_types).join(", ")}`);
				}

				if (valid_files.length > 0) {
					upload_component.load_files(valid_files);
				}
			} else {
				upload_component.load_files(files);
			}
		}
	}

	var div = root_1$1();
	let classes;
	var node = child(div);

	BlockTitle(node, {
		get show_label() {
			return show_label();
		},

		get info() {
			return info();
		},

		get rtl() {
			return rtl();
		},

		children: ($$anchor, $$slotProps) => {
			next();

			var text_1 = text();

			template_effect(() => set_text(text_1, $$props.label));
			append($$anchor, text_1);
		},
		$$slots: { default: true }
	});

	var div_1 = sibling(node, 2);
	var node_1 = child(div_1);

	{
		var consequent_1 = ($$anchor) => {
			var div_2 = root_3();
			let classes_1;
			var node_2 = child(div_2);

			{
				var consequent = ($$anchor) => {
					var div_3 = root_4();
					var node_3 = child(div_3);

					{
						let $0 = user_derived(() => $$props.label || "Audio");

						MinimalAudioRecorder(node_3, {
							get label() {
								return get($0);
							},

							get waveform_settings() {
								return $$props.waveform_settings;
							},

							get recording() {
								return get(recording);
							},

							get upload() {
								return $$props.upload;
							},

							get root() {
								return $$props.root;
							},

							get max_file_size() {
								return max_file_size();
							},

							onchange: (audio_value) => {
								set(mic_audio, audio_value, true);
							},

							onstoprecording: () => {
								set(recording, false);
								$$props.onstop_recording?.();
							},

							onclear: () => {
								active_source(null);
								set(recording, false);
								set(mic_audio, null);
								$$props.onclear?.();
							},

							get upload_promise() {
								return upload_promise();
							},

							set upload_promise($$value) {
								upload_promise($$value);
							}
						});
					}

					reset(div_3);
					append($$anchor, div_3);
				};

				var alternate = ($$anchor) => {
					var div_4 = root_5();
					var node_4 = child(div_4);

					{
						let $0 = user_derived(() => $$props.label || "Audio");

						MinimalAudioPlayer(node_4, {
							get value() {
								return get(mic_audio);
							},

							get label() {
								return get($0);
							},
							loop: false
						});
					}

					var div_5 = sibling(node_4, 2);
					var button = child(div_5);

					button.__click = () => {
						add_mic_audio_to_files();
						active_source(null);
						set(recording, false);
					};

					var node_5 = child(button);

					Check(node_5);
					reset(button);

					var button_1 = sibling(button, 2);

					button_1.__click = () => {
						active_source(null);
						set(recording, false);
						set(mic_audio, null);
					};

					var node_6 = child(button_1);

					Clear(node_6);
					reset(button_1);
					reset(div_5);
					reset(div_4);
					append($$anchor, div_4);
				};

				if_block(node_2, ($$render) => {
					if (!get(mic_audio)) $$render(consequent); else $$render(alternate, false);
				});
			}

			reset(div_2);
			template_effect(() => classes_1 = set_class(div_2, 1, 'recording-overlay svelte-1qn0337', null, classes_1, { 'has-audio': get(mic_audio) !== null }));
			append($$anchor, div_2);
		};

		if_block(node_1, ($$render) => {
			if (get(sources) && get(sources).includes("microphone") && active_source() === "microphone") $$render(consequent_1);
		});
	}

	var node_7 = sibling(node_1, 2);

	{
		var consequent_2 = ($$anchor) => {
			bind_this(
				Upload($$anchor, {
					onload: handle_upload,
					get file_count() {
						return file_count();
					},

					get filetype() {
						return get(file_types);
					},

					get root() {
						return $$props.root;
					},

					get max_file_size() {
						return max_file_size();
					},
					show_progress: false,
					disable_click: true,
					get onerror() {
						return $$props.onerror;
					},
					hidden: true,
					get upload() {
						return $$props.upload;
					},

					get stream_handler() {
						return $$props.stream_handler;
					},

					get upload_promise() {
						return upload_promise();
					},

					set upload_promise($$value) {
						upload_promise($$value);
					},

					get dragging() {
						return dragging();
					},

					set dragging($$value) {
						dragging($$value);
					},

					get uploading() {
						return get(uploading);
					},

					set uploading($$value) {
						set(uploading, $$value, true);
					}
				}),
				($$value) => upload_component = $$value,
				() => upload_component
			);
		};

		if_block(node_7, ($$render) => {
			if (get(show_upload)) $$render(consequent_2);
		});
	}

	var div_6 = sibling(node_7, 2);
	let classes_2;
	var node_8 = child(div_6);

	{
		var consequent_8 = ($$anchor) => {
			var div_7 = root_7();
			var node_9 = child(div_7);

			{
				var consequent_3 = ($$anchor) => {
					var button_2 = root_8();

					button_2.__click = handle_upload_click;

					var node_10 = child(button_2);

					Paperclip(node_10);
					reset(button_2);
					template_effect(() => button_2.disabled = disabled());
					append($$anchor, button_2);
				};

				if_block(node_9, ($$render) => {
					if (get(show_upload)) $$render(consequent_3);
				});
			}

			var node_11 = sibling(node_9, 2);

			each(node_11, 17, () => value()?.files ?? [], index, ($$anchor, file, index) => {
				var span = root_9();
				var div_8 = child(span);
				var node_12 = child(div_8);

				{
					var consequent_4 = ($$anchor) => {
						Image($$anchor, {
							get src() {
								return get(file).url;
							},

							restProps: {
								title: null,
								alt: "",
								loading: "lazy",
								class: "thumbnail-image"
							}
						});
					};

					var alternate_3 = ($$anchor) => {
						var fragment_3 = comment();
						var node_13 = first_child(fragment_3);

						{
							var consequent_5 = ($$anchor) => {
								Music($$anchor);
							};

							var alternate_2 = ($$anchor) => {
								var fragment_5 = comment();
								var node_14 = first_child(fragment_5);

								{
									var consequent_6 = ($$anchor) => {
										Video($$anchor);
									};

									var alternate_1 = ($$anchor) => {
										File$1($$anchor);
									};

									if_block(
										node_14,
										($$render) => {
											if (get(file).mime_type && get(file).mime_type.includes("video")) $$render(consequent_6); else $$render(alternate_1, false);
										},
										true
									);
								}

								append($$anchor, fragment_5);
							};

							if_block(
								node_13,
								($$render) => {
									if (get(file).mime_type && get(file).mime_type.includes("audio")) $$render(consequent_5); else $$render(alternate_2, false);
								},
								true
							);
						}

						append($$anchor, fragment_3);
					};

					if_block(node_12, ($$render) => {
						if (get(file).mime_type && get(file).mime_type.includes("image")) $$render(consequent_4); else $$render(alternate_3, false);
					});
				}

				var button_3 = sibling(node_12, 2);

				button_3.__click = (event) => remove_thumbnail(event, index);

				var node_15 = child(button_3);

				Clear(node_15);
				reset(button_3);
				reset(div_8);
				reset(span);
				append($$anchor, span);
			});

			var node_16 = sibling(node_11, 2);

			{
				var consequent_7 = ($$anchor) => {
					var div_9 = root_16();

					append($$anchor, div_9);
				};

				if_block(node_16, ($$render) => {
					if (get(uploading)) $$render(consequent_7);
				});
			}

			reset(div_7);
			append($$anchor, div_7);
		};

		if_block(node_8, ($$render) => {
			if ((value()?.files?.length ?? 0) > 0 || get(uploading)) $$render(consequent_8);
		});
	}

	var div_10 = sibling(node_8, 2);
	var node_17 = child(div_10);

	{
		var consequent_9 = ($$anchor) => {
			var button_4 = root_17();

			button_4.__click = handle_upload_click;

			var node_18 = child(button_4);

			Paperclip(node_18);
			reset(button_4);
			template_effect(() => button_4.disabled = disabled());
			append($$anchor, button_4);
		};

		if_block(node_17, ($$render) => {
			if (get(show_upload) && (value()?.files?.length ?? 0) === 0 && !get(uploading)) $$render(consequent_9);
		});
	}

	var div_11 = sibling(node_17, 2);
	var textarea = child(div_11);

	remove_textarea_child(textarea);

	let classes_3;

	action(textarea, ($$node, $$action_arg) => text_area_resize?.($$node, $$action_arg), () => ({
		text: value().text,
		lines: lines(),
		max_lines: $$props.max_lines
	}));

	effect(() => bind_value(textarea, () => value().text, ($$value) => value(value().text = $$value, true)));
	bind_this(textarea, ($$value) => el = $$value, () => el);

	var node_19 = sibling(textarea, 2);

	ScrollFade(node_19, {
		get visible() {
			return get(show_fade);
		},
		position: 'absolute'
	});

	reset(div_11);

	var node_20 = sibling(div_11, 2);

	{
		var consequent_10 = ($$anchor) => {
			var button_5 = root_18();
			let classes_4;

			button_5.__click = async () => {
				if (active_source() !== "microphone") {
					active_source("microphone");
					await tick();
					set(recording, true);
				} else {
					active_source(null);
					set(recording, false);
				}
			};

			var node_21 = child(button_5);

			Microphone(node_21);
			reset(button_5);

			template_effect(() => {
				classes_4 = set_class(button_5, 1, 'microphone-button svelte-1qn0337', null, classes_4, { recording: get(recording) });
				button_5.disabled = disabled();
			});

			append($$anchor, button_5);
		};

		if_block(node_20, ($$render) => {
			if (get(sources) && get(sources).includes("microphone")) $$render(consequent_10);
		});
	}

	var node_22 = sibling(node_20, 2);

	{
		var consequent_12 = ($$anchor) => {
			var button_6 = root_19();
			let classes_5;

			button_6.__click = handle_submit;

			var node_23 = child(button_6);

			{
				var consequent_11 = ($$anchor) => {
					ArrowUp($$anchor);
				};

				var alternate_4 = ($$anchor) => {
					var text_2 = text();

					template_effect(() => set_text(text_2, submit_btn()));
					append($$anchor, text_2);
				};

				if_block(node_23, ($$render) => {
					if (submit_btn() === true) $$render(consequent_11); else $$render(alternate_4, false);
				});
			}

			reset(button_6);

			template_effect(() => {
				classes_5 = set_class(button_6, 1, 'submit-button svelte-1qn0337', null, classes_5, { 'padded-button': submit_btn() !== true });
				button_6.disabled = disabled();
			});

			append($$anchor, button_6);
		};

		if_block(node_22, ($$render) => {
			if (submit_btn()) $$render(consequent_12);
		});
	}

	var node_24 = sibling(node_22, 2);

	{
		var consequent_14 = ($$anchor) => {
			var button_7 = root_22();
			let classes_6;

			button_7.__click = handle_stop;

			var node_25 = child(button_7);

			{
				var consequent_13 = ($$anchor) => {
					Square($$anchor, { fill: "none", stroke_width: 2.5 });
				};

				var alternate_5 = ($$anchor) => {
					var text_3 = text();

					template_effect(() => set_text(text_3, stop_btn()));
					append($$anchor, text_3);
				};

				if_block(node_25, ($$render) => {
					if (stop_btn() === true) $$render(consequent_13); else $$render(alternate_5, false);
				});
			}

			reset(button_7);
			template_effect(() => classes_6 = set_class(button_7, 1, 'stop-button svelte-1qn0337', null, classes_6, { 'padded-button': stop_btn() !== true }));
			append($$anchor, button_7);
		};

		if_block(node_24, ($$render) => {
			if (stop_btn()) $$render(consequent_14);
		});
	}

	reset(div_10);
	reset(div_6);
	reset(div_1);
	reset(div);
	bind_this(div, ($$value) => full_container = $$value, () => full_container);

	template_effect(() => {
		classes = set_class(div, 1, 'full-container svelte-1qn0337', null, classes, { dragging: dragging() });

		classes_2 = set_class(div_6, 1, 'input-wrapper svelte-1qn0337', null, classes_2, {
			'has-files': (value()?.files?.length ?? 0) > 0 || get(uploading)
		});

		set_attribute(textarea, 'dir', rtl() ? "rtl" : "ltr");
		set_attribute(textarea, 'placeholder', placeholder());
		set_attribute(textarea, 'rows', lines());
		textarea.disabled = disabled();
		set_style(textarea, text_align() ? "text-align: " + text_align() : "");
		set_attribute(textarea, 'autocapitalize', html_attributes()?.autocapitalize);
		set_attribute(textarea, 'autocorrect', html_attributes()?.autocorrect);
		set_attribute(textarea, 'spellcheck', html_attributes()?.spellcheck);
		set_attribute(textarea, 'autocomplete', html_attributes()?.autocomplete);
		set_attribute(textarea, 'tabindex', html_attributes()?.tabindex);
		set_attribute(textarea, 'enterkeyhint', html_attributes()?.enterkeyhint);
		set_attribute(textarea, 'lang', html_attributes()?.lang);
		classes_3 = set_class(textarea, 1, 'svelte-1qn0337', null, classes_3, { 'no-label': !show_label() });
		textarea.dir = textarea.dir;
	});

	event('dragenter', div, handle_dragenter);
	event('dragleave', div, handle_dragleave);
	event('dragover', div, (e) => e.preventDefault());
	event('drop', div, handle_drop);
	event('keypress', textarea, handle_keypress);
	event('blur', textarea, () => $$props.onblur?.());
	event('select', textarea, handle_select);
	event('focus', textarea, () => $$props.onfocus?.());
	event('scroll', textarea, handle_scroll);
	event('paste', textarea, handle_paste);
	append($$anchor, div);
	pop();
}

delegate(['click']);

var root_1 = from_html(`<!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	let upload_promise = state(null);

	class MultimodalTextboxGradio extends Gradio {
		async get_data() {
			if (get(upload_promise)) {
				await get(upload_promise);
				await tick();
			}

			const data = await super.get_data();

			return data;
		}
	}

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	const gradio = new MultimodalTextboxGradio(props);

	gradio.props.value = gradio.props.value ?? { text: "", files: [] };

	let dragging = state(false);
	let active_source = state(null);
	let color_accent = "darkorange";

	const waveform_settings = {
		height: 50,
		barWidth: 2,
		barGap: 3,
		cursorWidth: 2,
		cursorColor: "#ddd5e9",
		autoplay: false,
		barRadius: 10,
		dragToSeek: true,
		normalize: true,
		minPxPerSec: 20,
		waveColor: "",
		progressColor: "",
		mediaControls: false,
		sampleRate: 44100
	};

	onMount(() => {
		color_accent = getComputedStyle(document?.documentElement).getPropertyValue("--color-accent");
		set_trim_region_colour();
		waveform_settings.waveColor = gradio.props?.waveform_options?.waveform_color || "#9ca3af";
		waveform_settings.progressColor = gradio.props?.waveform_options?.waveform_progress_color || color_accent;
		waveform_settings.mediaControls = gradio.props?.waveform_options?.show_controls;
		waveform_settings.sampleRate = gradio.props?.waveform_options?.sample_rate || 44100;
	});

	const trim_region_settings = {
		color: gradio.props?.waveform_options?.trim_region_color};

	function set_trim_region_colour() {
		document.documentElement.style.setProperty("--trim-region-color", trim_region_settings.color || color_accent);
	}

	// Create const references to the callbacks so that afterUpdate in child is not called on every prop change
	// in the DOM. See https://github.com/gradio-app/gradio/issues/11933
	// eslint-disable-next-line @typescript-eslint/explicit-function-return-type
	const upload_fn = (...args) => gradio.shared.client.upload(...args);

	const i18n = (s) => gradio.i18n(s);
	const stream_handler_fn = (...args) => gradio.shared.client.stream(...args);
	let sources_string = user_derived(() => gradio.props.sources.join(","));
	let file_types_string = user_derived(() => (gradio.props.file_types || []).join(",") || null);

	{
		let $0 = user_derived(() => [...gradio.shared.elem_classes || [], "multimodal-textbox"]);
		let $1 = user_derived(() => get(dragging) ? "focus" : "base");

		Block($$anchor, {
			get visible() {
				return gradio.shared.visible;
			},

			get elem_id() {
				return gradio.shared.elem_id;
			},

			get elem_classes() {
				return get($0);
			},

			get scale() {
				return gradio.shared.scale;
			},

			get min_width() {
				return gradio.shared.min_width;
			},
			allow_overflow: false,
			padding: false,
			get border_mode() {
				return get($1);
			},

			get rtl() {
				return gradio.props.rtl;
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

				{
					let $0 = user_derived(() => gradio.shared.label || "MultimodalTextbox");
					let $1 = user_derived(() => !gradio.props.max_lines ? gradio.props.lines + 1 : gradio.props.max_lines);
					let $2 = user_derived(() => !gradio.shared.interactive);

					MultimodalTextbox(node_1, {
						value_is_output: true,
						get file_types_string() {
							return get(file_types_string);
						},

						get root() {
							return gradio.shared.root;
						},

						get label() {
							return get($0);
						},

						get info() {
							return gradio.props.info;
						},

						get show_label() {
							return gradio.shared.show_label;
						},

						get lines() {
							return gradio.props.lines;
						},

						get rtl() {
							return gradio.props.rtl;
						},

						get text_align() {
							return gradio.props.text_align;
						},

						get waveform_settings() {
							return waveform_settings;
						},
						i18n,
						get max_lines() {
							return get($1);
						},

						get placeholder() {
							return gradio.props.placeholder;
						},

						get submit_btn() {
							return gradio.props.submit_btn;
						},

						get stop_btn() {
							return gradio.props.stop_btn;
						},

						get autofocus() {
							return gradio.props.autofocus;
						},

						get autoscroll() {
							return gradio.shared.autoscroll;
						},

						get file_count() {
							return gradio.props.file_count;
						},

						get sources_string() {
							return get(sources_string);
						},

						get max_file_size() {
							return gradio.shared.max_file_size;
						},

						onchange: (e) => {
							gradio.props.value = e;
							gradio.dispatch("change", gradio.props.value);
						},
						oninput: () => gradio.dispatch("input"),
						onsubmit: () => gradio.dispatch("submit"),
						onstop: () => gradio.dispatch("stop"),
						onblur: () => gradio.dispatch("blur"),
						onselect: (e) => gradio.dispatch("select", e),
						onfocus: () => gradio.dispatch("focus"),
						onerror: (detail) => {
							gradio.dispatch("error", detail);
						},
						onstop_recording: () => gradio.dispatch("stop_recording"),
						onupload: (e) => gradio.dispatch("upload", e),
						onclear: () => gradio.dispatch("clear"),
						get disabled() {
							return get($2);
						},
						upload: upload_fn,
						stream_handler: stream_handler_fn,
						get max_plain_text_length() {
							return gradio.props.max_plain_text_length;
						},

						get html_attributes() {
							return gradio.props.html_attributes;
						},

						get upload_promise() {
							return get(upload_promise);
						},

						set upload_promise($$value) {
							set(upload_promise, $$value, true);
						},

						get value() {
							return gradio.props.value;
						},

						set value($$value) {
							gradio.props.value = $$value;
						},

						get dragging() {
							return get(dragging);
						},

						set dragging($$value) {
							set(dragging, $$value, true);
						},

						get active_source() {
							return get(active_source);
						},

						set active_source($$value) {
							set(active_source, $$value, true);
						}
					});
				}

				append($$anchor, fragment_1);
			},
			$$slots: { default: true }
		});
	}

	pop();
}

export { MultimodalTextbox as BaseMultimodalTextbox, Index as default };
//# sourceMappingURL=Index-DiVjWtzo.js.map
