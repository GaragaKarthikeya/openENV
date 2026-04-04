import { e as init, f as set_style, p as prop, d as bind_this, i as if_block, b as set_class, a as set_attribute, k as each, L as set_selected, u as index } from './i18n-CMo5lpzy.js';
import { N as teardown, aM as get_descriptor, a as append, f as from_svg, R as push, a4 as createEventDispatcher, V as child, Y as reset, X as sibling, Z as event, T as pop, W as from_html, aa as onMount, ae as onDestroy, t as template_effect, U as flushSync, w as get, a3 as mutable_source, x as set, G as deep_read_state, z as untrack, au as prepare_files, S as first_child, a6 as comment, A as effect, a0 as set_text } from './index-BvBk1Iap.js';
import { t as transition, f as fade } from './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import { a as action } from './actions-Cit_UA5S.js';
import { D as DropdownArrow } from './DropdownArrow-Dw1NZKCM.js';
import { W as Webcam$1 } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { S as Square } from './Square-B6INQV79.js';
import { S as Spinner } from './Spinner-BrK-6emq.js';
import { S as StreamingBar } from './StreamingBar-bVF1kkqe.js';
/* empty css                                                     */

/**
 * Makes an `export`ed (non-prop) variable available on the `$$props` object
 * so that consumers can do `bind:x` on the component.
 * @template V
 * @param {Record<string, unknown>} props
 * @param {string} prop
 * @param {V} value
 * @returns {void}
 */
function bind_prop(props, prop, value) {
	var desc = get_descriptor(props, prop);

	if (desc && desc.set) {
		props[prop] = value;
		teardown(() => {
			props[prop] = null;
		});
	}
}

var root$2 = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="feather feather-camera"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>`);

function Camera($$anchor) {
	var svg = root$2();

	append($$anchor, svg);
}

var root$1 = from_svg(`<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="feather feather-circle"><circle cx="12" cy="12" r="10"></circle></svg>`);

function Circle($$anchor) {
	var svg = root$1();

	append($$anchor, svg);
}

var root = from_html(`<button class="svelte-1dqolfz"><div class="wrap svelte-1dqolfz"><span class="icon-wrap svelte-1dqolfz"><!></span> </div></button>`);

function WebcamPermissions($$anchor, $$props) {
	push($$props, false);

	const dispatch = createEventDispatcher();

	init();

	var button = root();

	set_style(button, '', {}, { height: '100%' });

	var div = child(button);
	var span = child(div);
	var node = child(span);

	Webcam$1(node);
	reset(span);

	var text = sibling(span);

	text.nodeValue = ' Click to Access Webcam';
	reset(div);
	reset(button);
	event('click', button, () => dispatch("click"));
	append($$anchor, button);
	pop();
}

function get_devices() {
  return navigator.mediaDevices.enumerateDevices();
}
async function set_local_stream(local_stream, video_source) {
  video_source.srcObject = local_stream;
  video_source.muted = true;
  await video_source.play();
}
async function get_video_stream(include_audio, video_source, webcam_constraints, device_id) {
  const constraints = {
    video: device_id ? { deviceId: { exact: device_id }, ...webcam_constraints?.video } : webcam_constraints?.video || {
      width: { ideal: 1920 },
      height: { ideal: 1440 }
    },
    audio: include_audio && (webcam_constraints?.audio ?? true)
    // Defaults to true if not specified
  };
  return navigator.mediaDevices.getUserMedia(constraints).then((local_stream) => {
    set_local_stream(local_stream, video_source);
    return local_stream;
  });
}
function set_available_devices(devices) {
  const cameras = devices.filter(
    (device) => device.kind === "videoinput"
  );
  return cameras;
}

var root_2 = from_html(`<div title="grant webcam access" style="height: 100%"><!></div>`);
var root_5 = from_html(`<div class="icon-with-text svelte-1tktvmr" style="width:var(--size-24);"><div class="icon color-primary svelte-1tktvmr" title="spinner"><!></div> </div>`);
var root_7 = from_html(`<div class="icon-with-text svelte-1tktvmr"><div class="icon color-primary svelte-1tktvmr" title="stop recording"><!></div> </div>`);
var root_8 = from_html(`<div class="icon-with-text svelte-1tktvmr"><div class="icon color-primary svelte-1tktvmr" title="start recording"><!></div> </div>`);
var root_9 = from_html(`<div class="icon svelte-1tktvmr" title="capture photo"><!></div>`);
var root_10 = from_html(`<button class="icon svelte-1tktvmr" aria-label="select input source"><!></button>`);
var root_12 = from_html(`<option class="svelte-1tktvmr"> </option>`);
var root_14 = from_html(`<option class="svelte-1tktvmr"> </option>`);
var root_11 = from_html(`<select class="select-wrap svelte-1tktvmr" aria-label="select source"><!></select>`);
var root_3 = from_html(`<div class="button-wrap svelte-1tktvmr"><button class="svelte-1tktvmr"><!></button> <!></div> <!>`, 1);
var root_1 = from_html(`<div class="wrap svelte-1tktvmr"><!>  <video></video> <img/> <!></div>`, 2);

function Webcam($$anchor, $$props) {
	push($$props, false);

	let video_source = mutable_source();
	let available_video_devices = mutable_source([]);
	let selected_device = mutable_source(null);
	let stream_state = prop($$props, 'stream_state', 12, "closed");
	let canvas;
	let streaming = prop($$props, 'streaming', 12, false);
	let pending = prop($$props, 'pending', 12, false);
	let root = prop($$props, 'root', 12, "");
	let stream_every = prop($$props, 'stream_every', 12, 1);
	let mode = prop($$props, 'mode', 12, "image");
	let mirror_webcam = prop($$props, 'mirror_webcam', 12);
	let include_audio = prop($$props, 'include_audio', 12);
	let webcam_constraints = prop($$props, 'webcam_constraints', 12, null);
	let i18n = prop($$props, 'i18n', 12);
	let upload = prop($$props, 'upload', 12);
	let value = prop($$props, 'value', 12, null);
	let time_limit = prop($$props, 'time_limit', 12, null);
	const dispatch = createEventDispatcher();

	onMount(() => {
		canvas = document.createElement("canvas");

		if (streaming() && mode() === "image") {
			window.setInterval(
				() => {
					if (get(video_source) && !pending()) {
						take_picture();
					}
				},
				stream_every() * 1000
			);
		}
	});

	const handle_device_change = async (event) => {
		const target = event.target;
		const device_id = target.value;

		await get_video_stream(include_audio(), get(video_source), webcam_constraints(), device_id).then(async (local_stream) => {
			stream = local_stream;
			set(selected_device, get(available_video_devices).find((device) => device.deviceId === device_id) || null);
			set(options_open, false);
		});
	};

	async function access_webcam() {
		try {
			get_video_stream(include_audio(), get(video_source), webcam_constraints()).then(async (local_stream) => {
				set(webcam_accessed, true);
				set(available_video_devices, await get_devices());
				stream = local_stream;
			}).then(() => set_available_devices(get(available_video_devices))).then((devices) => {
				set(available_video_devices, devices);

				const used_devices = stream.getTracks().map((track) => track.getSettings()?.deviceId)[0];

				set(selected_device, used_devices
					? devices.find((device) => device.deviceId === used_devices) || get(available_video_devices)[0]
					: get(available_video_devices)[0]);
			});

			if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
				dispatch("error", i18n()("image.no_webcam_support"));
			}
		} catch(err) {
			if (err instanceof DOMException && err.name == "NotAllowedError") {
				dispatch("error", i18n()("image.allow_webcam_access"));
			} else {
				throw err;
			}
		}
	}

	function take_picture() {
		if ((!streaming() || streaming() && get(recording)) && get(video_source).videoWidth && get(video_source).videoHeight) {
			var context = canvas.getContext("2d");

			canvas.width = get(video_source).videoWidth;
			canvas.height = get(video_source).videoHeight;
			context.drawImage(get(video_source), 0, 0, get(video_source).videoWidth, get(video_source).videoHeight);

			if (mirror_webcam()) {
				context.scale(-1, 1);
				context.drawImage(get(video_source), -get(video_source).videoWidth, 0);
			}

			if (streaming() && (!get(recording) || stream_state() === "waiting")) {
				return;
			}

			if (streaming()) {
				const image_data = canvas.toDataURL("image/jpeg");

				dispatch("stream", image_data);

				return;
			}

			canvas.toBlob(
				(blob) => {
					dispatch(streaming() ? "stream" : "capture", blob);
				},
				`image/${streaming() ? "jpeg" : "png"}`,
				0.8
			);
		}
	}

	let recording = mutable_source(false);
	let recorded_blobs = [];
	let stream;
	let mimeType;
	let media_recorder;

	function take_recording() {
		if (get(recording)) {
			media_recorder.stop();

			let video_blob = new Blob(recorded_blobs, { type: mimeType });
			let ReaderObj = new FileReader();

			ReaderObj.onload = async function (e) {
				if (e.target) {
					let _video_blob = new File([video_blob], "sample." + mimeType.substring(6));
					const val = await prepare_files([_video_blob]);
					let val_ = (await upload()(val, root()))?.filter(Boolean)[0];

					dispatch("capture", val_);
					dispatch("stop_recording");
				}
			};

			ReaderObj.readAsDataURL(video_blob);
		} else if (typeof MediaRecorder !== "undefined") {
			dispatch("start_recording");
			recorded_blobs = [];

			let validMimeTypes = ["video/webm", "video/mp4"];

			for (let validMimeType of validMimeTypes) {
				if (MediaRecorder.isTypeSupported(validMimeType)) {
					mimeType = validMimeType;

					break;
				}
			}

			if (mimeType === null) {
				console.error("No supported MediaRecorder mimeType");

				return;
			}

			media_recorder = new MediaRecorder(stream, { mimeType });

			media_recorder.addEventListener("dataavailable", function (e) {
				recorded_blobs.push(e.data);
			});

			media_recorder.start(200);
		}

		set(recording, !get(recording));
	}

	let webcam_accessed = mutable_source(false);

	function record_video_or_photo({ destroy } = {}) {
		if (mode() === "image" && streaming()) {
			set(recording, !get(recording));
		}

		if (!destroy) {
			if (mode() === "image") {
				take_picture();
			} else {
				take_recording();
			}
		}

		if (!get(recording) && stream) {
			dispatch("close_stream");
		}
	}

	let options_open = mutable_source(false);

	function click_outside(node, cb) {
		const handle_click = (event) => {
			if (node && !node.contains(event.target) && !event.defaultPrevented) {
				cb(event);
			}
		};

		document.addEventListener("click", handle_click, true);

		return {
			destroy() {
				document.removeEventListener("click", handle_click, true);
			}
		};
	}

	function handle_click_outside(event) {
		event.preventDefault();
		event.stopPropagation();
		set(options_open, false);
	}

	onDestroy(() => {
		if (typeof window === "undefined") return;

		record_video_or_photo({ destroy: true });
		stream?.getTracks().forEach((track) => track.stop());
	});

	var $$exports = {
		click_outside,
		get stream_state() {
			return stream_state();
		},

		set stream_state($$value) {
			stream_state($$value);
			flushSync();
		},

		get streaming() {
			return streaming();
		},

		set streaming($$value) {
			streaming($$value);
			flushSync();
		},

		get pending() {
			return pending();
		},

		set pending($$value) {
			pending($$value);
			flushSync();
		},

		get root() {
			return root();
		},

		set root($$value) {
			root($$value);
			flushSync();
		},

		get stream_every() {
			return stream_every();
		},

		set stream_every($$value) {
			stream_every($$value);
			flushSync();
		},

		get mode() {
			return mode();
		},

		set mode($$value) {
			mode($$value);
			flushSync();
		},

		get mirror_webcam() {
			return mirror_webcam();
		},

		set mirror_webcam($$value) {
			mirror_webcam($$value);
			flushSync();
		},

		get include_audio() {
			return include_audio();
		},

		set include_audio($$value) {
			include_audio($$value);
			flushSync();
		},

		get webcam_constraints() {
			return webcam_constraints();
		},

		set webcam_constraints($$value) {
			webcam_constraints($$value);
			flushSync();
		},

		get i18n() {
			return i18n();
		},

		set i18n($$value) {
			i18n($$value);
			flushSync();
		},

		get upload() {
			return upload();
		},

		set upload($$value) {
			upload($$value);
			flushSync();
		},

		get value() {
			return value();
		},

		set value($$value) {
			value($$value);
			flushSync();
		},

		get time_limit() {
			return time_limit();
		},

		set time_limit($$value) {
			time_limit($$value);
			flushSync();
		}
	};

	init();

	var div = root_1();
	var node_1 = child(div);

	StreamingBar(node_1, {
		get time_limit() {
			return time_limit();
		}
	});

	var video = sibling(node_1, 2);
	let classes;

	bind_this(video, ($$value) => set(video_source, $$value), () => get(video_source));

	var img = sibling(video, 2);
	let classes_1;
	var node_2 = sibling(img, 2);

	{
		var consequent = ($$anchor) => {
			var div_1 = root_2();
			var node_3 = child(div_1);

			WebcamPermissions(node_3, { $$events: { click: async () => access_webcam() } });
			reset(div_1);
			transition(1, div_1, () => fade, () => ({ delay: 100, duration: 200 }));
			append($$anchor, div_1);
		};

		var alternate_4 = ($$anchor) => {
			var fragment = root_3();
			var div_2 = first_child(fragment);
			var button = child(div_2);
			var node_4 = child(button);

			{
				var consequent_3 = ($$anchor) => {
					var fragment_1 = comment();
					var node_5 = first_child(fragment_1);

					{
						var consequent_1 = ($$anchor) => {
							var div_3 = root_5();
							var div_4 = child(div_3);
							var node_6 = child(div_4);

							Spinner(node_6);
							reset(div_4);

							var text = sibling(div_4);

							reset(div_3);

							template_effect(($0) => set_text(text, ` ${$0 ?? ''}`), [
								() => (
									deep_read_state(i18n()),
									untrack(() => i18n()("audio.waiting"))
								)
							]);

							append($$anchor, div_3);
						};

						var alternate_1 = ($$anchor) => {
							var fragment_2 = comment();
							var node_7 = first_child(fragment_2);

							{
								var consequent_2 = ($$anchor) => {
									var div_5 = root_7();
									var div_6 = child(div_5);
									var node_8 = child(div_6);

									Square(node_8, {});
									reset(div_6);

									var text_1 = sibling(div_6);

									reset(div_5);

									template_effect(($0) => set_text(text_1, ` ${$0 ?? ''}`), [
										() => (
											deep_read_state(i18n()),
											untrack(() => i18n()("audio.stop"))
										)
									]);

									append($$anchor, div_5);
								};

								var alternate = ($$anchor) => {
									var div_7 = root_8();
									var div_8 = child(div_7);
									var node_9 = child(div_8);

									Circle(node_9);
									reset(div_8);

									var text_2 = sibling(div_8);

									reset(div_7);

									template_effect(($0) => set_text(text_2, ` ${$0 ?? ''}`), [
										() => (
											deep_read_state(i18n()),
											untrack(() => i18n()("audio.record"))
										)
									]);

									append($$anchor, div_7);
								};

								if_block(
									node_7,
									($$render) => {
										if (streaming() && stream_state() === "open" || !streaming() && get(recording)) $$render(consequent_2); else $$render(alternate, false);
									},
									true
								);
							}

							append($$anchor, fragment_2);
						};

						if_block(node_5, ($$render) => {
							if (streaming() && stream_state() === "waiting") $$render(consequent_1); else $$render(alternate_1, false);
						});
					}

					append($$anchor, fragment_1);
				};

				var alternate_2 = ($$anchor) => {
					var div_9 = root_9();
					var node_10 = child(div_9);

					Camera(node_10);
					reset(div_9);
					append($$anchor, div_9);
				};

				if_block(node_4, ($$render) => {
					if (mode() === "video" || streaming()) $$render(consequent_3); else $$render(alternate_2, false);
				});
			}

			reset(button);

			var node_11 = sibling(button, 2);

			{
				var consequent_4 = ($$anchor) => {
					var button_1 = root_10();
					var node_12 = child(button_1);

					DropdownArrow(node_12);
					reset(button_1);
					event('click', button_1, () => set(options_open, true));
					append($$anchor, button_1);
				};

				if_block(node_11, ($$render) => {
					if (!get(recording)) $$render(consequent_4);
				});
			}

			reset(div_2);

			var node_13 = sibling(div_2, 2);

			{
				var consequent_6 = ($$anchor) => {
					var select = root_11();
					var node_14 = child(select);

					{
						var consequent_5 = ($$anchor) => {
							var option = root_12();
							var text_3 = child(option, true);

							reset(option);
							option.value = option.__value = '';

							template_effect(($0) => set_text(text_3, $0), [
								() => (
									deep_read_state(i18n()),
									untrack(() => i18n()("common.no_devices"))
								)
							]);

							append($$anchor, option);
						};

						var alternate_3 = ($$anchor) => {
							var fragment_3 = comment();
							var node_15 = first_child(fragment_3);

							each(node_15, 1, () => get(available_video_devices), index, ($$anchor, device) => {
								var option_1 = root_14();
								var text_4 = child(option_1, true);

								reset(option_1);

								var option_1_value = {};

								template_effect(() => {
									set_selected(option_1, (
										get(selected_device),
										get(device),
										untrack(() => get(selected_device).deviceId === get(device).deviceId)
									));

									set_text(text_4, (get(device), untrack(() => get(device).label)));

									if (option_1_value !== (option_1_value = (get(device), untrack(() => get(device).deviceId)))) {
										option_1.value = (option_1.__value = (get(device), untrack(() => get(device).deviceId))) ?? '';
									}
								});

								append($$anchor, option_1);
							});

							append($$anchor, fragment_3);
						};

						if_block(node_14, ($$render) => {
							if ((
								get(available_video_devices),
								untrack(() => get(available_video_devices).length === 0)
							)) $$render(consequent_5); else $$render(alternate_3, false);
						});
					}

					reset(select);
					action(select, ($$node, $$action_arg) => click_outside?.($$node, $$action_arg), () => handle_click_outside);
					effect(() => event('change', select, handle_device_change));
					append($$anchor, select);
				};

				if_block(node_13, ($$render) => {
					if (get(options_open) && get(selected_device)) $$render(consequent_6);
				});
			}

			template_effect(() => set_attribute(button, 'aria-label', mode() === "image" ? "capture photo" : "start recording"));
			event('click', button, () => record_video_or_photo());
			append($$anchor, fragment);
		};

		if_block(node_2, ($$render) => {
			if (!get(webcam_accessed)) $$render(consequent); else $$render(alternate_4, false);
		});
	}

	reset(div);

	template_effect(() => {
		classes = set_class(video, 1, 'svelte-1tktvmr', null, classes, {
			flip: mirror_webcam(),
			hide: !get(webcam_accessed) || get(webcam_accessed) && !!value()
		});

		set_attribute(img, 'src', (deep_read_state(value()), untrack(() => value()?.url)));

		classes_1 = set_class(img, 1, 'svelte-1tktvmr', null, classes_1, {
			hide: !get(webcam_accessed) || get(webcam_accessed) && !value()
		});
	});

	append($$anchor, div);
	bind_prop($$props, 'click_outside', click_outside);

	return pop($$exports);
}

export { Circle as C, Webcam as W, bind_prop as b };
//# sourceMappingURL=Webcam-rWLs1V82.js.map
