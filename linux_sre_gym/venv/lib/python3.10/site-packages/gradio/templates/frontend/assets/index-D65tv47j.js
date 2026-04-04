const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./module-C16d28tm.js","./module-DYU7z__K.js","./module-Bkn9j7N6.js"])))=>i.map(i=>d[i]);
import { p as prop, i as if_block, k as each, u as index, d as bind_this, b as set_class, f as set_style, h as clsx, a as set_attribute, r as rest_props, g as spread_props } from './i18n-CMo5lpzy.js';
import { R as push, aa as onMount, V as child, Y as reset, t as template_effect, a as append, T as pop, a0 as set_text, a6 as comment, S as first_child, w as get, W as from_html, aj as delegate, u as state, v as proxy, ae as onDestroy, y as user_effect, x as set, X as sibling, aN as update, a5 as user_derived, af as __vitePreload, au as prepare_files, M as tick } from './index-BvBk1Iap.js';
import { f as format_time, u as uploadToHuggingFace, G as Gradio } from './utils.svelte-CYLB_jIA.js';
import { S as Static } from './index-Lz7fsHFt.js';
import './StreamingBar.svelte_svelte_type_style_lang-CdFIGsJa.js';
import { W as WaveformControls, A as AudioPlayer, S as StaticAudio } from './StaticAudio-DGaZb-fW.js';
import { s as snippet } from './snippet-Bl5YVoCg.js';
import { U as Upload } from './Upload-CPeZiqam.js';
import { M as ModifyUpload } from './ModifyUpload-B5K-5Hqx.js';
import { S as SelectSource } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { B as BlockLabel } from './BlockLabel-oGSqUii_.js';
import { S as ShareButton } from './ShareButton-Dz0y1C1K.js';
import { M as Music } from './Music-D0NN2hV9.js';
import { C as CustomButton } from './IconButtonWrapper-Ba0GidgU.js';
import { S as StreamingBar } from './StreamingBar-bVF1kkqe.js';
import { s as skip_audio, u, p as process_audio } from './MinimalAudioPlayer-Cdqg8MIC.js';
import { o } from './record.esm-BVY9RaAH.js';
import { P as Pause } from './VolumeLevels-DHAc0pUL.js';
import { S as Spinner } from './Spinner-BrK-6emq.js';
import { B as Block } from './Block-Bx3BikCN.js';
import { U as UploadText } from './UploadText-BSD1IoCI.js';
export { default as BaseExample } from './Example-hTHSlRG1.js';
import './clone-DaB-S2nH.js';
import './Clear-Dgs5UhTO.js';
import './html-CdUEtR5E.js';
import './index-C7e-J7CF.js';
import './DownloadLink-KvrFKyMn.js';
import './Empty-CHa-gxv7.js';
import './Download-pFhUaTl4.js';
import './Play-CWcFSzc7.js';
import './Undo-BxcJDM8E.js';
import './actions-Cit_UA5S.js';
import './Edit-Dr58YXBX.js';

var root_1$4 = from_html(`<option> </option>`);
var root_3$2 = from_html(`<option> </option>`);
var root$3 = from_html(`<select class="mic-select svelte-ym1wxn" aria-label="Select input device"><!></select>`);

function DeviceSelect($$anchor, $$props) {
	push($$props, true);

	let micDevices = prop($$props, 'micDevices', 15);

	onMount(() => {
		if (typeof window !== "undefined" && navigator.mediaDevices) {
			let tempDevices = [];

			o.getAvailableAudioDevices().then((devices) => {
				micDevices(devices);

				devices.forEach((device) => {
					if (device.deviceId) {
						tempDevices.push(device);
					}
				});

				micDevices(tempDevices);
			}).catch((err) => {
				if (err instanceof DOMException && err.name == "NotAllowedError") {
					$$props.onerror?.($$props.i18n("audio.allow_recording_access"));
				}
			});
		}
	});

	var select = root$3();
	var node = child(select);

	{
		var consequent = ($$anchor) => {
			var option = root_1$4();
			var text = child(option, true);

			reset(option);
			option.value = option.__value = '';
			template_effect(($0) => set_text(text, $0), [() => $$props.i18n("audio.no_microphone")]);
			append($$anchor, option);
		};

		var alternate = ($$anchor) => {
			var fragment = comment();
			var node_1 = first_child(fragment);

			each(node_1, 17, micDevices, index, ($$anchor, micDevice) => {
				var option_1 = root_3$2();
				var text_1 = child(option_1, true);

				reset(option_1);

				var option_1_value = {};

				template_effect(() => {
					set_text(text_1, get(micDevice).label);

					if (option_1_value !== (option_1_value = get(micDevice).deviceId)) {
						option_1.value = (option_1.__value = get(micDevice).deviceId) ?? '';
					}
				});

				append($$anchor, option_1);
			});

			append($$anchor, fragment);
		};

		if_block(node, ($$render) => {
			if (!micDevices() || micDevices().length === 0) $$render(consequent); else $$render(alternate, false);
		});
	}

	reset(select);
	template_effect(() => select.disabled = !micDevices() || micDevices().length === 0);
	append($$anchor, select);
	pop();
}

var root_1$3 = from_html(`<time class="duration-button duration svelte-1xuh0j1"> </time>`);
var root$2 = from_html(`<div class="controls svelte-1xuh0j1"><div class="wrapper svelte-1xuh0j1"><button class="record record-button svelte-1xuh0j1"> </button> <button> </button> <button id="stop-paused" class="stop-button-paused svelte-1xuh0j1"> </button> <button aria-label="pause" class="pause-button svelte-1xuh0j1"><!></button> <button class="resume-button svelte-1xuh0j1"> </button> <!></div> <!></div>`);

function WaveformRecordControls($$anchor, $$props) {
	push($$props, true);

	let recording = prop($$props, 'recording', 3, false),
		timing = prop($$props, 'timing', 3, false);

	let micDevices = state(proxy([]));
	let recordButton;
	let pauseButton;
	let resumeButton;
	let stopButton;
	let stopButtonPaused;
	let recording_ongoing = state(false);

	const handleRecordStart = () => {
		recordButton.style.display = "none";
		stopButton.style.display = "flex";
		pauseButton.style.display = "block";
	};

	const handleRecordEnd = () => {
		if ($$props.record.isPaused()) {
			$$props.record.resumeRecording();
			$$props.record.stopRecording();
		}

		$$props.record.stopMic();
		recordButton.style.display = "flex";
		stopButton.style.display = "none";
		pauseButton.style.display = "none";
		recordButton.disabled = false;
	};

	const handleRecordPause = () => {
		pauseButton.style.display = "none";
		resumeButton.style.display = "block";
		stopButton.style.display = "none";
		stopButtonPaused.style.display = "flex";
	};

	const handleRecordResume = () => {
		pauseButton.style.display = "block";
		resumeButton.style.display = "none";
		recordButton.style.display = "none";
		stopButton.style.display = "flex";
		stopButtonPaused.style.display = "none";
	};

	onMount(() => {
		$$props.record.on("record-start", handleRecordStart);
		$$props.record.on("record-end", handleRecordEnd);
		$$props.record.on("record-pause", handleRecordPause);
		$$props.record.on("record-resume", handleRecordResume);
	});

	onDestroy(() => {
		$$props.record.un("record-start", handleRecordStart);
		$$props.record.un("record-end", handleRecordEnd);
		$$props.record.un("record-pause", handleRecordPause);
		$$props.record.un("record-resume", handleRecordResume);
	});

	user_effect(() => {
		if (recording() && !get(recording_ongoing)) {
			$$props.record.startMic().then(() => {
				$$props.record.startRecording();
				set(recording_ongoing, true);
			});
		} else if (!recording() && get(recording_ongoing)) {
			if ($$props.record.isPaused()) {
				$$props.record.resumeRecording();
			}

			$$props.record.stopRecording();
			set(recording_ongoing, false);
		}
	});

	var div = root$2();
	var div_1 = child(div);
	var button = child(div_1);

	button.__click = () => $$props.record.startRecording();

	var text = child(button, true);

	reset(button);
	bind_this(button, ($$value) => recordButton = $$value, () => recordButton);

	var button_1 = sibling(button, 2);

	button_1.__click = () => {
		if ($$props.record.isPaused()) {
			$$props.record.resumeRecording();
			$$props.record.stopRecording();
		}

		$$props.record.stopRecording();
	};

	var text_1 = child(button_1, true);

	reset(button_1);
	bind_this(button_1, ($$value) => stopButton = $$value, () => stopButton);

	var button_2 = sibling(button_1, 2);

	button_2.__click = () => {
		if ($$props.record.isPaused()) {
			$$props.record.resumeRecording();
			$$props.record.stopRecording();
		}

		$$props.record.stopRecording();
	};

	var text_2 = child(button_2, true);

	reset(button_2);
	bind_this(button_2, ($$value) => stopButtonPaused = $$value, () => stopButtonPaused);

	var button_3 = sibling(button_2, 2);

	button_3.__click = () => $$props.record.pauseRecording();

	var node = child(button_3);

	Pause(node);
	reset(button_3);
	bind_this(button_3, ($$value) => pauseButton = $$value, () => pauseButton);

	var button_4 = sibling(button_3, 2);

	button_4.__click = () => $$props.record.resumeRecording();

	var text_3 = child(button_4, true);

	reset(button_4);
	bind_this(button_4, ($$value) => resumeButton = $$value, () => resumeButton);

	var node_1 = sibling(button_4, 2);

	{
		var consequent = ($$anchor) => {
			var time = root_1$3();
			var text_4 = child(time, true);

			reset(time);
			template_effect(() => set_text(text_4, $$props.record_time));
			append($$anchor, time);
		};

		if_block(node_1, ($$render) => {
			if (timing() && !$$props.show_recording_waveform) $$render(consequent);
		});
	}

	reset(div_1);

	var node_2 = sibling(div_1, 2);

	DeviceSelect(node_2, {
		get i18n() {
			return $$props.i18n;
		},

		get micDevices() {
			return get(micDevices);
		},

		set micDevices($$value) {
			set(micDevices, $$value, true);
		}
	});

	reset(div);

	template_effect(
		($0, $1, $2, $3, $4) => {
			set_text(text, $0);
			set_class(button_1, 1, `stop-button ${$1 ?? ''}`, 'svelte-1xuh0j1');
			set_text(text_1, $2);
			set_text(text_2, $3);
			set_text(text_3, $4);
		},
		[
			() => $$props.i18n("audio.record"),
			() => $$props.record.isPaused() ? 'stop-button-paused' : '',
			() => $$props.i18n("audio.stop"),
			() => $$props.i18n("audio.stop"),
			() => $$props.i18n("audio.resume")
		]
	);

	append($$anchor, div);
	pop();
}

delegate(['click']);

var root_2$2 = from_html(`<time class="trim-duration svelte-j9q3sk"> </time>`);
var root_3$1 = from_html(`<time class="duration svelte-j9q3sk"> </time>`);
var root_4$2 = from_html(`<time class="duration svelte-j9q3sk">0:00</time>`);
var root_1$2 = from_html(`<div class="timestamps svelte-j9q3sk"><time class="time svelte-j9q3sk">0:00</time> <div><!> <!></div></div>`);
var root$1 = from_html(`<div class="component-wrapper svelte-j9q3sk"><div class="microphone svelte-j9q3sk" data-testid="microphone-waveform"></div> <div data-testid="recording-waveform"></div> <!> <!> <!></div>`);

function AudioRecorder($$anchor, $$props) {
	push($$props, true);

	let mode = prop($$props, 'mode', 15),
		waveform_options = prop($$props, 'waveform_options', 19, () => ({ show_recording_waveform: true })),
		editable = prop($$props, 'editable', 3, true),
		recording = prop($$props, 'recording', 3, false);

	let micWaveform;
	let recordingWaveform = state(undefined);
	let playing = state(false);
	let recordingContainer;
	let microphoneContainer;
	let record = state(undefined);
	let recordedAudio = state(null);

	// timestamps
	let timeRef;

	let durationRef;
	let audio_duration = state(0);
	let seconds = state(0);
	let interval;
	let timing = state(false);

	// trimming
	let trimDuration = state(0);

	let record_mounted = state(false);

	const start_interval = () => {
		clearInterval(interval);

		interval = setInterval(
			() => {
				update(seconds);
			},
			1000
		);
	};

	function record_start_callback() {
		start_interval();
		set(timing, true);
		$$props.onstartrecording?.();

		if (waveform_options().show_recording_waveform) {
			let waveformCanvas = microphoneContainer;

			if (waveformCanvas) waveformCanvas.style.display = "block";
		}
	}

	async function record_end_callback(blob) {
		set(seconds, 0);
		set(timing, false);
		clearInterval(interval);

		try {
			const array_buffer = await blob.arrayBuffer();
			const context = new AudioContext({ sampleRate: $$props.waveform_settings.sampleRate });
			const audio_buffer = await context.decodeAudioData(array_buffer);

			if (audio_buffer) await process_audio(audio_buffer).then(async (audio) => {
				await $$props.dispatch_blob([audio], "change");
				await $$props.dispatch_blob([audio], "stop_recording");
			});
		} catch(e) {
			console.error(e);
		}
	}

	user_effect(() => {
		get(record)?.on("record-resume", () => {
			start_interval();
		});
	});

	user_effect(() => {
		get(recordingWaveform)?.on("decode", (duration) => {
			set(audio_duration, duration, true);
			durationRef && (durationRef.textContent = format_time(duration));
		});
	});

	user_effect(() => {
		get(recordingWaveform)?.on("timeupdate", (currentTime) => timeRef && (timeRef.textContent = format_time(currentTime)));
	});

	user_effect(() => {
		get(recordingWaveform)?.on("pause", () => {
			$$props.onpause?.();
			set(playing, false);
		});
	});

	user_effect(() => {
		get(recordingWaveform)?.on("play", () => {
			$$props.onplay?.();
			set(playing, true);
		});
	});

	user_effect(() => {
		get(recordingWaveform)?.on("finish", () => {
			$$props.onstop?.();
			set(playing, false);
		});
	});

	const create_mic_waveform = () => {
		if (microphoneContainer) microphoneContainer.innerHTML = "";
		if (micWaveform !== undefined) micWaveform.destroy();
		if (!microphoneContainer) return;

		micWaveform = u.create({
			...$$props.waveform_settings,
			normalize: false,
			container: microphoneContainer
		});

		set(record, micWaveform.registerPlugin(o.create()), true);
		get(record)?.on("record-end", record_end_callback);
		get(record)?.on("record-start", record_start_callback);

		get(record)?.on("record-pause", () => {
			$$props.onpauserecording?.();
			clearInterval(interval);
		});

		get(record)?.on("record-end", (blob) => {
			set(recordedAudio, URL.createObjectURL(blob), true);

			const microphone = microphoneContainer;
			const recording = recordingContainer;

			if (microphone) microphone.style.display = "none";

			if (recording && get(recordedAudio)) {
				recording.innerHTML = "";
				create_recording_waveform();
			}
		});

		set(record_mounted, true);
	};

	const create_recording_waveform = () => {
		let recording = recordingContainer;

		if (!get(recordedAudio) || !recording) return;

		set(
			recordingWaveform,
			u.create({
				container: recording,
				url: get(recordedAudio),
				...$$props.waveform_settings
			}),
			true
		);
	};

	const handle_trim_audio = async (start, end) => {
		mode("edit");

		const decodedData = get(recordingWaveform)?.getDecodedData();

		if (decodedData) await process_audio(decodedData, start, end).then(async (trimmedAudio) => {
			await $$props.dispatch_blob([trimmedAudio], "change");
			await $$props.dispatch_blob([trimmedAudio], "stop_recording");
			get(recordingWaveform)?.destroy();
			create_recording_waveform();
		});

		$$props.onedit?.();
	};

	onMount(() => {
		create_mic_waveform();

		window.addEventListener("keydown", (e) => {
			const is_focused_in_waveform = recordingContainer && recordingContainer.contains(document.activeElement);

			if (!is_focused_in_waveform) return;

			if (e.key === "ArrowRight") {
				skip_audio(get(recordingWaveform), 0.1);
			} else if (e.key === "ArrowLeft") {
				skip_audio(get(recordingWaveform), -0.1);
			}
		});
	});

	var div = root$1();
	var div_1 = child(div);

	bind_this(div_1, ($$value) => microphoneContainer = $$value, () => microphoneContainer);

	var div_2 = sibling(div_1, 2);

	bind_this(div_2, ($$value) => recordingContainer = $$value, () => recordingContainer);

	var node = sibling(div_2, 2);

	{
		var consequent_2 = ($$anchor) => {
			var div_3 = root_1$2();
			var time = child(div_3);

			bind_this(time, ($$value) => timeRef = $$value, () => timeRef);

			var div_4 = sibling(time, 2);
			var node_1 = child(div_4);

			{
				var consequent = ($$anchor) => {
					var time_1 = root_2$2();
					var text = child(time_1, true);

					reset(time_1);
					template_effect(($0) => set_text(text, $0), [() => format_time(get(trimDuration))]);
					append($$anchor, time_1);
				};

				if_block(node_1, ($$render) => {
					if (mode() === "edit" && get(trimDuration) > 0) $$render(consequent);
				});
			}

			var node_2 = sibling(node_1, 2);

			{
				var consequent_1 = ($$anchor) => {
					var time_2 = root_3$1();
					var text_1 = child(time_2, true);

					reset(time_2);
					template_effect(($0) => set_text(text_1, $0), [() => format_time(get(seconds))]);
					append($$anchor, time_2);
				};

				var alternate = ($$anchor) => {
					var time_3 = root_4$2();

					bind_this(time_3, ($$value) => durationRef = $$value, () => durationRef);
					append($$anchor, time_3);
				};

				if_block(node_2, ($$render) => {
					if (get(timing)) $$render(consequent_1); else $$render(alternate, false);
				});
			}

			reset(div_4);
			reset(div_3);
			append($$anchor, div_3);
		};

		if_block(node, ($$render) => {
			if ((get(timing) || get(recordedAudio)) && waveform_options().show_recording_waveform) $$render(consequent_2);
		});
	}

	var node_3 = sibling(node, 2);

	{
		var consequent_3 = ($$anchor) => {
			{
				let $0 = user_derived(() => format_time(get(seconds)));

				WaveformRecordControls($$anchor, {
					get record() {
						return get(record);
					},

					get i18n() {
						return $$props.i18n;
					},

					get timing() {
						return get(timing);
					},

					get recording() {
						return recording();
					},

					get show_recording_waveform() {
						return waveform_options().show_recording_waveform;
					},

					get record_time() {
						return get($0);
					}
				});
			}
		};

		if_block(node_3, ($$render) => {
			if (get(record_mounted) && !get(recordedAudio)) $$render(consequent_3);
		});
	}

	var node_4 = sibling(node_3, 2);

	{
		var consequent_4 = ($$anchor) => {
			WaveformControls($$anchor, {
				get waveform() {
					return get(recordingWaveform);
				},

				get container() {
					return recordingContainer;
				},

				get playing() {
					return get(playing);
				},

				get audio_duration() {
					return get(audio_duration);
				},

				get i18n() {
					return $$props.i18n;
				},

				get editable() {
					return editable();
				},
				interactive: true,
				handle_trim_audio,
				show_redo: true,
				get handle_reset_value() {
					return $$props.handle_reset_value;
				},

				get waveform_options() {
					return waveform_options();
				},

				get trimDuration() {
					return get(trimDuration);
				},

				set trimDuration($$value) {
					set(trimDuration, $$value, true);
				},

				get mode() {
					return mode();
				},

				set mode($$value) {
					mode($$value);
				}
			});
		};

		if_block(node_4, ($$render) => {
			if (get(recordingWaveform) && get(recordedAudio)) $$render(consequent_4);
		});
	}

	reset(div);
	append($$anchor, div);
	pop();
}

var root_1$1 = from_html(`<div class="svelte-m6ymia"></div>`);
var root_2$1 = from_html(`<button><span class="record-icon svelte-m6ymia"><span class="pinger svelte-m6ymia"></span> <span class="dot svelte-m6ymia"></span></span> </button>`);
var root_4$1 = from_html(`<button class="spinner-button svelte-m6ymia"><div class="icon svelte-m6ymia"><!></div> </button>`);
var root_5 = from_html(`<button class="record-button svelte-m6ymia"><span class="record-icon svelte-m6ymia"><span class="dot svelte-m6ymia"></span></span> </button>`);
var root = from_html(`<div class="mic-wrap svelte-m6ymia"><!> <div class="controls svelte-m6ymia"><!> <!></div></div>`);

function StreamAudio($$anchor, $$props) {
	push($$props, true);

	let recording = prop($$props, 'recording', 3, false),
		paused_recording = prop($$props, 'paused_recording', 3, false),
		waveform_options = prop($$props, 'waveform_options', 19, () => ({ show_recording_waveform: true })),
		waiting = prop($$props, 'waiting', 3, false);

	let micWaveform;
	let waveformRecord;
	let microphoneContainer;
	let micDevices = state(proxy([]));

	onMount(() => {
		create_mic_waveform();
	});

	const create_mic_waveform = () => {
		if (micWaveform !== undefined) micWaveform.destroy();
		if (!microphoneContainer) return;

		micWaveform = u.create({
			...$$props.waveform_settings,
			normalize: false,
			container: microphoneContainer
		});

		waveformRecord = micWaveform.registerPlugin(o.create());
	};

	var div = root();
	var node = child(div);

	{
		var consequent = ($$anchor) => {
			var div_1 = root_1$1();
			let styles;

			bind_this(div_1, ($$value) => microphoneContainer = $$value, () => microphoneContainer);
			template_effect(() => styles = set_style(div_1, '', styles, { display: recording() ? "block" : "none" }));
			append($$anchor, div_1);
		};

		if_block(node, ($$render) => {
			if (waveform_options().show_recording_waveform) $$render(consequent);
		});
	}

	var div_2 = sibling(node, 2);
	var node_1 = child(div_2);

	{
		var consequent_1 = ($$anchor) => {
			var button = root_2$1();

			button.__click = () => {
				waveformRecord?.stopMic();
				$$props.stop();
			};

			var text = sibling(child(button));

			reset(button);

			template_effect(
				($0) => {
					set_class(button, 1, clsx(paused_recording() ? "stop-button-paused" : "stop-button"), 'svelte-m6ymia');
					set_text(text, ` ${$0 ?? ''}`);
				},
				[
					() => paused_recording()
						? $$props.i18n("audio.pause")
						: $$props.i18n("audio.stop")
				]
			);

			append($$anchor, button);
		};

		var alternate_1 = ($$anchor) => {
			var fragment = comment();
			var node_2 = first_child(fragment);

			{
				var consequent_2 = ($$anchor) => {
					var button_1 = root_4$1();

					button_1.__click = () => {
						$$props.stop();
					};

					var div_3 = child(button_1);
					var node_3 = child(div_3);

					Spinner(node_3);
					reset(div_3);

					var text_1 = sibling(div_3);

					reset(button_1);
					template_effect(($0) => set_text(text_1, ` ${$0 ?? ''}`), [() => $$props.i18n("audio.waiting")]);
					append($$anchor, button_1);
				};

				var alternate = ($$anchor) => {
					var button_2 = root_5();

					button_2.__click = () => {
						waveformRecord?.startMic();
						$$props.record();
					};

					var text_2 = sibling(child(button_2));

					reset(button_2);
					template_effect(($0) => set_text(text_2, ` ${$0 ?? ''}`), [() => $$props.i18n("audio.record")]);
					append($$anchor, button_2);
				};

				if_block(
					node_2,
					($$render) => {
						if (recording() && waiting()) $$render(consequent_2); else $$render(alternate, false);
					},
					true
				);
			}

			append($$anchor, fragment);
		};

		if_block(node_1, ($$render) => {
			if (recording() && !waiting()) $$render(consequent_1); else $$render(alternate_1, false);
		});
	}

	var node_4 = sibling(node_1, 2);

	DeviceSelect(node_4, {
		get i18n() {
			return $$props.i18n;
		},

		get micDevices() {
			return get(micDevices);
		},

		set micDevices($$value) {
			set(micDevices, $$value, true);
		}
	});

	reset(div_2);
	reset(div);
	append($$anchor, div);
	pop();
}

delegate(['click']);

let media_recorder;
async function init_media_recorder() {
  const { MediaRecorder, register } = await __vitePreload(async () => { const { MediaRecorder, register } = await import('./module-C16d28tm.js');return { MediaRecorder, register }},true              ?__vite__mapDeps([0,1]):void 0,import.meta.url);
  const { connect } = await __vitePreload(async () => { const { connect } = await import('./module-Bkn9j7N6.js');return { connect }},true              ?__vite__mapDeps([2,1]):void 0,import.meta.url);
  register(await connect());
  media_recorder = MediaRecorder;
  return media_recorder;
}

var root_3 = from_html(`<!> <!>`, 1);
var root_10 = from_html(`<!> <!>`, 1);
var root_1 = from_html(`<!> <div><!> <!> <!></div>`, 1);

function InteractiveAudio($$anchor, $$props) {
	push($$props, true);

	let value = prop($$props, 'value', 15, null),
		subtitles = prop($$props, 'subtitles', 3, null),
		show_label = prop($$props, 'show_label', 3, true),
		buttons = prop($$props, 'buttons', 19, () => ["download", "share"]),
		on_custom_button_click = prop($$props, 'on_custom_button_click', 3, null),
		sources = prop($$props, 'sources', 19, () => ["microphone", "upload"]),
		pending = prop($$props, 'pending', 3, false),
		streaming = prop($$props, 'streaming', 3, false),
		trim_region_settings = prop($$props, 'trim_region_settings', 19, () => ({})),
		waveform_options = prop($$props, 'waveform_options', 19, () => ({})),
		dragging = prop($$props, 'dragging', 15, false),
		active_source = prop($$props, 'active_source', 15, "microphone"),
		handle_reset_value = prop($$props, 'handle_reset_value', 3, () => {}),
		editable = prop($$props, 'editable', 3, true),
		max_file_size = prop($$props, 'max_file_size', 3, null),
		stream_every = prop($$props, 'stream_every', 3, 0.1),
		uploading = prop($$props, 'uploading', 15, false),
		recording = prop($$props, 'recording', 15, false),
		class_name = prop($$props, 'class_name', 3, ""),
		upload_promise = prop($$props, 'upload_promise', 15),
		initial_value = prop($$props, 'initial_value', 15),
		playback_position = prop($$props, 'playback_position', 15),
		time_limit = prop($$props, 'time_limit', 3, null),
		stream_state = prop($$props, 'stream_state', 3, "closed");

	user_effect(() => {
		$$props.ondrag?.(dragging());
	});

	// TODO: make use of this
	// export let type: "normal" | "numpy" = "normal";
	let recorder;

	let mode = state("");
	let header = undefined;
	let pending_stream = [];
	let submit_pending_stream_on_pending_end = false;
	let inited = false;
	let streaming_media_recorder;
	const NUM_HEADER_BYTES = 44;
	let audio_chunks = [];
	const is_browser = typeof window !== "undefined";

	if (is_browser && streaming()) {
		init_media_recorder().then((a) => {
			streaming_media_recorder = a;
		});
	}

	const to_blob_parts = (parts) => parts.map((part) => {
		if (part instanceof Blob) return part;

		return part.slice();
	});

	const dispatch_blob = async (blobs, event) => {
		let _audio_blob = new File(to_blob_parts(blobs), "audio.wav", { type: "audio/wav" });

		if (_audio_blob.size === 0) {
			return;
		}

		const val = await prepare_files([_audio_blob], event === "stream");

		initial_value(value());
		value((await $$props.upload(val, $$props.root, undefined, max_file_size() || undefined))?.filter(Boolean)[0]);

		if (event === "stream") {
			$$props.onstream?.(value());
		} else if (event === "change") {
			$$props.onchange?.(value());
		} else if (event === "stop_recording") {
			$$props.onstop_recording?.();
		}
	};

	onDestroy(() => {
		if (streaming() && recorder && recorder.state !== "inactive") {
			recorder.stop();
		}
	});

	async function prepare_audio() {
		let stream;

		try {
			stream = await navigator.mediaDevices.getUserMedia({ audio: true });
		} catch(err) {
			if (!navigator.mediaDevices) {
				$$props.onerror?.($$props.i18n("audio.no_device_support"));

				return;
			}

			if (err instanceof DOMException && err.name == "NotAllowedError") {
				$$props.onerror?.($$props.i18n("audio.allow_recording_access"));

				return;
			}

			throw err;
		}

		if (stream == null) return;

		if (streaming()) {
			recorder = new streaming_media_recorder(stream, { mimeType: "audio/wav" });
			recorder.addEventListener("dataavailable", handle_chunk);
		} else {
			recorder = new MediaRecorder(stream);

			recorder.addEventListener("dataavailable", (event) => {
				audio_chunks.push(event.data);
			});
		}

		recorder.addEventListener("stop", async () => {
			recording(false);
			recorder.stop();
			await dispatch_blob(audio_chunks, "change");
			await dispatch_blob(audio_chunks, "stop_recording");
			audio_chunks = [];
		});

		inited = true;
	}

	async function handle_chunk(event) {
		let buffer = await event.data.arrayBuffer();
		let payload = new Uint8Array(buffer);

		if (!header) {
			header = new Uint8Array(buffer.slice(0, NUM_HEADER_BYTES));
			payload = new Uint8Array(buffer.slice(NUM_HEADER_BYTES));
		}

		if (pending()) {
			pending_stream.push(payload);
		} else {
			let blobParts = [header].concat(pending_stream, [payload]);

			if (!recording() || stream_state() === "waiting") return;

			dispatch_blob(blobParts, "stream");
			pending_stream = [];
		}
	}

	user_effect(() => {
		if (submit_pending_stream_on_pending_end && pending() === false) {
			submit_pending_stream_on_pending_end = false;

			if (header && pending_stream) {
				let blobParts = [header].concat(pending_stream);

				pending_stream = [];
				dispatch_blob(blobParts, "stream");
			}
		}
	});

	async function record() {
		recording(true);
		$$props.onstart_recording?.();

		if (!inited) await prepare_audio();

		header = undefined;

		if (streaming() && recorder.state != "recording") {
			recorder.start(stream_every() * 1000);
		}
	}

	function clear() {
		$$props.onchange?.(null);
		$$props.onclear?.();
		set(mode, "");
		value(null);
	}

	function handle_load(detail) {
		value(detail);
		$$props.onchange?.(detail);
		$$props.onupload?.(detail);
	}

	async function stop() {
		recording(false);

		if (streaming()) {
			$$props.onclose_stream?.();
			$$props.onstop_recording?.();
			recorder.stop();

			if (pending()) {
				submit_pending_stream_on_pending_end = true;
			}

			await dispatch_blob(audio_chunks, "stop_recording");
			$$props.onclear?.();
			set(mode, "");
		}
	}

	user_effect(() => {
		if (!recording() && recorder && get(mode) !== "") stop();
	});

	user_effect(() => {
		if (recording() && recorder) record();
	});

	var fragment = root_1();
	var node = first_child(fragment);

	{
		let $0 = user_derived(() => active_source() === "upload" && value() === null);
		let $1 = user_derived(() => $$props.label || $$props.i18n("audio.audio"));

		BlockLabel(node, {
			get show_label() {
				return show_label();
			},

			get Icon() {
				return Music;
			},

			get float() {
				return get($0);
			},

			get label() {
				return get($1);
			}
		});
	}

	var div = sibling(node, 2);
	var node_1 = child(div);

	StreamingBar(node_1, {
		get time_limit() {
			return time_limit();
		}
	});

	var node_2 = sibling(node_1, 2);

	{
		var consequent_4 = ($$anchor) => {
			var fragment_1 = comment();
			var node_3 = first_child(fragment_1);

			{
				var consequent_1 = ($$anchor) => {
					var fragment_2 = root_3();
					var node_4 = first_child(fragment_2);

					ModifyUpload(node_4, {
						get i18n() {
							return $$props.i18n;
						},
						onclear: clear
					});

					var node_5 = sibling(node_4, 2);

					{
						var consequent = ($$anchor) => {
							{
								let $0 = user_derived(() => stream_state() === "waiting");

								StreamAudio($$anchor, {
									record,
									get recording() {
										return recording();
									},
									stop,
									get i18n() {
										return $$props.i18n;
									},

									get waveform_settings() {
										return $$props.waveform_settings;
									},

									get waveform_options() {
										return waveform_options();
									},

									get waiting() {
										return get($0);
									}
								});
							}
						};

						var alternate = ($$anchor) => {
							AudioRecorder($$anchor, {
								get i18n() {
									return $$props.i18n;
								},

								get editable() {
									return editable();
								},

								get recording() {
									return recording();
								},
								dispatch_blob,
								get waveform_settings() {
									return $$props.waveform_settings;
								},

								get waveform_options() {
									return waveform_options();
								},

								get handle_reset_value() {
									return handle_reset_value();
								},
								onstartrecording: () => $$props.onstart_recording?.(),
								onpauserecording: () => $$props.onpause_recording?.(),
								onstoprecording: () => $$props.onstop_recording?.(),
								get mode() {
									return get(mode);
								},

								set mode($$value) {
									set(mode, $$value, true);
								}
							});
						};

						if_block(node_5, ($$render) => {
							if (streaming()) $$render(consequent); else $$render(alternate, false);
						});
					}

					append($$anchor, fragment_2);
				};

				var alternate_1 = ($$anchor) => {
					var fragment_5 = comment();
					var node_6 = first_child(fragment_5);

					{
						var consequent_3 = ($$anchor) => {
							{
								let $0 = user_derived(() => $$props.i18n("audio.drop_to_upload"));

								Upload($$anchor, {
									filetype: 'audio/aac,audio/midi,audio/mpeg,audio/ogg,audio/wav,audio/x-wav,audio/opus,audio/webm,audio/flac,audio/vnd.rn-realaudio,audio/x-ms-wma,audio/x-aiff,audio/amr,audio/*',
									onload: handle_load,
									onerror: (detail) => $$props.onerror?.(detail),
									get root() {
										return $$props.root;
									},

									get max_file_size() {
										return max_file_size();
									},

									get upload() {
										return $$props.upload;
									},

									get stream_handler() {
										return $$props.stream_handler;
									},

									get aria_label() {
										return get($0);
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
										return uploading();
									},

									set uploading($$value) {
										uploading($$value);
									},

									children: ($$anchor, $$slotProps) => {
										var fragment_7 = comment();
										var node_7 = first_child(fragment_7);

										{
											var consequent_2 = ($$anchor) => {
												var fragment_8 = comment();
												var node_8 = first_child(fragment_8);

												snippet(node_8, () => $$props.children);
												append($$anchor, fragment_8);
											};

											if_block(node_7, ($$render) => {
												if ($$props.children) $$render(consequent_2);
											});
										}

										append($$anchor, fragment_7);
									},
									$$slots: { default: true }
								});
							}
						};

						if_block(
							node_6,
							($$render) => {
								if (active_source() === "upload") $$render(consequent_3);
							},
							true
						);
					}

					append($$anchor, fragment_5);
				};

				if_block(node_3, ($$render) => {
					if (active_source() === "microphone") $$render(consequent_1); else $$render(alternate_1, false);
				});
			}

			append($$anchor, fragment_1);
		};

		var alternate_3 = ($$anchor) => {
			var fragment_9 = root_10();
			var node_9 = first_child(fragment_9);

			{
				let $0 = user_derived(() => buttons() === null
					? value().url
					: buttons().some((btn) => typeof btn === "string" && btn === "download") ? value().url : null);

				ModifyUpload(node_9, {
					get i18n() {
						return $$props.i18n;
					},
					onclear: clear,
					onedit: () => {
						set(mode, "edit");
						$$props.onedit?.();
					},

					get download() {
						return get($0);
					},

					children: ($$anchor, $$slotProps) => {
						var fragment_10 = comment();
						var node_10 = first_child(fragment_10);

						{
							var consequent_7 = ($$anchor) => {
								var fragment_11 = comment();
								var node_11 = first_child(fragment_11);

								each(node_11, 17, buttons, index, ($$anchor, btn) => {
									var fragment_12 = comment();
									var node_12 = first_child(fragment_12);

									{
										var consequent_6 = ($$anchor) => {
											var fragment_13 = comment();
											var node_13 = first_child(fragment_13);

											{
												var consequent_5 = ($$anchor) => {
													ShareButton($$anchor, {
														get i18n() {
															return $$props.i18n;
														},

														get onerror() {
															return $$props.onerror;
														},
														onshare: () => {},
														formatter: async (fileData) => {
															if (!fileData || !fileData.url) return "";

															let url = await uploadToHuggingFace(fileData.url);

															return `<audio controls src="${url}"></audio>`;
														},

														get value() {
															return value();
														}
													});
												};

												if_block(node_13, ($$render) => {
													if (get(btn) === "share") $$render(consequent_5);
												});
											}

											append($$anchor, fragment_13);
										};

										var alternate_2 = ($$anchor) => {
											CustomButton($$anchor, {
												get button() {
													return get(btn);
												},

												on_click: (id) => {
													if (on_custom_button_click()) {
														on_custom_button_click()(id);
													}
												}
											});
										};

										if_block(node_12, ($$render) => {
											if (typeof get(btn) === "string") $$render(consequent_6); else $$render(alternate_2, false);
										});
									}

									append($$anchor, fragment_12);
								});

								append($$anchor, fragment_11);
							};

							if_block(node_10, ($$render) => {
								if (value() !== null && buttons()) $$render(consequent_7);
							});
						}

						append($$anchor, fragment_10);
					},
					$$slots: { default: true }
				});
			}

			var node_14 = sibling(node_9, 2);

			{
				let $0 = user_derived(() => Array.isArray(subtitles()) ? subtitles() : subtitles()?.url);

				AudioPlayer(node_14, {
					get value() {
						return value();
					},

					get subtitles() {
						return get($0);
					},

					get label() {
						return $$props.label;
					},

					get i18n() {
						return $$props.i18n;
					},
					dispatch_blob,
					get waveform_settings() {
						return $$props.waveform_settings;
					},

					get waveform_options() {
						return waveform_options();
					},

					get trim_region_settings() {
						return trim_region_settings();
					},

					get handle_reset_value() {
						return handle_reset_value();
					},

					get editable() {
						return editable();
					},

					get loop() {
						return $$props.loop;
					},
					interactive: true,
					get onstop() {
						return $$props.onstop;
					},

					get onplay() {
						return $$props.onplay;
					},

					get onpause() {
						return $$props.onpause;
					},

					get onedit() {
						return $$props.onedit;
					},

					get mode() {
						return get(mode);
					},

					set mode($$value) {
						set(mode, $$value, true);
					},

					get playback_position() {
						return playback_position();
					},

					set playback_position($$value) {
						playback_position($$value);
					}
				});
			}

			append($$anchor, fragment_9);
		};

		if_block(node_2, ($$render) => {
			if (value() == null || streaming()) $$render(consequent_4); else $$render(alternate_3, false);
		});
	}

	var node_15 = sibling(node_2, 2);

	SelectSource(node_15, {
		get sources() {
			return sources();
		},
		handle_clear: clear,
		get active_source() {
			return active_source();
		},

		set active_source($$value) {
			active_source($$value);
		}
	});

	reset(div);

	template_effect(() => {
		set_class(div, 1, `audio-container ${class_name() ?? ''}`, 'svelte-ocxd3m');
		set_attribute(div, 'data-testid', $$props.label ? "waveform-" + $$props.label : "unlabelled-audio");
	});

	append($$anchor, fragment);
	pop();
}

var root_2 = from_html(`<!> <!>`, 1);
var root_4 = from_html(`<!> <!>`, 1);

function Index($$anchor, $$props) {
	push($$props, true);

	let props = rest_props($$props, ['$$slots', '$$events', '$$legacy']);
	let upload_promise = state(void 0);

	$$props.props.stream_every = 0.1; // default to 0.1s stream interval

	class AudioGradio extends Gradio {
		async get_data() {
			if (get(upload_promise)) {
				await get(upload_promise);
				await tick();
			}

			const data = await super.get_data();

			return data;
		}
	}

	const gradio = new AudioGradio(props);
	let label = user_derived(() => gradio.shared.label || gradio.i18n("audio.audio"));
	let minimal = user_derived(() => $$props.minimal ?? gradio.props.minimal ?? false);
	let active_source = user_derived(() => gradio.props.sources ? gradio.props.sources[0] : null);
	let initial_value = state(proxy(gradio.props.value));

	const handle_reset_value = () => {
		if (get(initial_value) === null || gradio.props.value === get(initial_value)) {
			return;
		}

		gradio.props.value = get(initial_value);
	};

	let dragging = state(false);
	let recording = state(proxy(gradio.props.recording ?? false));

	user_effect(() => {
		gradio.props.recording = get(recording);
	});

	let color_accent = "darkorange";

	let waveform_settings = proxy({
		height: 50,
		barWidth: 2,
		barGap: 3,
		cursorWidth: 2,
		cursorColor: "#ddd5e9",
		autoplay: gradio.props.autoplay,
		barRadius: 10,
		dragToSeek: true,
		normalize: true,
		minPxPerSec: 20,
		waveColor: gradio.props.waveform_options.waveform_color || "#9ca3af",
		progressColor: gradio.props.waveform_options.waveform_progress_color || color_accent,
		mediaControls: gradio.props.waveform_options.show_controls ?? false,
		sampleRate: gradio.props.waveform_options.sample_rate || 44100
	});

	const trim_region_settings = {
		color: gradio.props.waveform_options.trim_region_color,
		drag: true,
		resize: true
	};

	function set_trim_region_colour() {
		document.documentElement.style.setProperty("--trim-region-color", trim_region_settings.color || color_accent);
	}

	function handle_error(detail) {
		const [level, status] = detail.includes("Invalid file type") ? ["warning", "complete"] : ["error", "error"];

		if (gradio.shared.loading_status) {
			gradio.shared.loading_status.status = status;
			gradio.shared.loading_status.message = detail;
		}

		gradio.dispatch(level, detail);
	}

	gradio.watch_for_change();

	onMount(() => {
		set_trim_region_colour();
	});

	var fragment = comment();
	var node = first_child(fragment);

	{
		var consequent = ($$anchor) => {
			{
				let $0 = user_derived(() => get(dragging) ? "focus" : "base");

				Block($$anchor, {
					variant: "solid",
					get border_mode() {
						return get($0);
					},
					padding: false,
					allow_overflow: false,
					get elem_id() {
						return gradio.shared.elem_id;
					},

					get elem_classes() {
						return gradio.shared.elem_classes;
					},

					get visible() {
						return gradio.shared.visible;
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

					children: ($$anchor, $$slotProps) => {
						var fragment_2 = root_2();
						var node_1 = first_child(fragment_2);

						Static(node_1, spread_props(
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

						var node_2 = sibling(node_1, 2);

						{
							let $0 = user_derived(() => gradio.props.buttons ?? ["download", "share"]);

							StaticAudio(node_2, {
								get i18n() {
									return gradio.i18n;
								},

								get show_label() {
									return gradio.shared.show_label;
								},

								get buttons() {
									return get($0);
								},

								get value() {
									return gradio.props.value;
								},

								get subtitles() {
									return gradio.props.subtitles;
								},

								get label() {
									return get(label);
								},

								get loop() {
									return gradio.props.loop;
								},

								get waveform_settings() {
									return waveform_settings;
								},

								get waveform_options() {
									return gradio.props.waveform_options;
								},

								get editable() {
									return gradio.props.editable;
								},

								get minimal() {
									return get(minimal);
								},

								on_custom_button_click: (id) => {
									gradio.dispatch("custom_button_click", { id });
								},
								onshare: (detail) => gradio.dispatch("share", detail),
								onerror: (e) => gradio.dispatch("error", e.detail),
								onplay: () => gradio.dispatch("play"),
								onpause: () => gradio.dispatch("pause"),
								onstop: () => gradio.dispatch("stop"),
								get playback_position() {
									return gradio.props.playback_position;
								},

								set playback_position($$value) {
									gradio.props.playback_position = $$value;
								}
							});
						}

						append($$anchor, fragment_2);
					},
					$$slots: { default: true }
				});
			}
		};

		var alternate = ($$anchor) => {
			{
				let $0 = user_derived(() => gradio.props.value === null && get(active_source) === "upload" ? "dashed" : "solid");
				let $1 = user_derived(() => get(dragging) ? "focus" : "base");

				Block($$anchor, {
					get variant() {
						return get($0);
					},

					get border_mode() {
						return get($1);
					},
					padding: false,
					allow_overflow: false,
					get elem_id() {
						return gradio.shared.elem_id;
					},

					get elem_classes() {
						return gradio.shared.elem_classes;
					},

					get visible() {
						return gradio.shared.visible;
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

					children: ($$anchor, $$slotProps) => {
						var fragment_4 = root_4();
						var node_3 = first_child(fragment_4);

						Static(node_3, spread_props(
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

						var node_4 = sibling(node_3, 2);

						{
							let $0 = user_derived(() => gradio.props.buttons ?? []);
							let $1 = user_derived(() => get(active_source) || undefined);

							InteractiveAudio(node_4, {
								get label() {
									return get(label);
								},

								get show_label() {
									return gradio.shared.show_label;
								},

								get buttons() {
									return get($0);
								},

								on_custom_button_click: (id) => {
									gradio.dispatch("custom_button_click", { id });
								},

								get value() {
									return gradio.props.value;
								},

								get subtitles() {
									return gradio.props.subtitles;
								},
								onchange: (detail) => gradio.props.value = detail,
								onstream: (detail) => {
									gradio.props.value = detail;
									gradio.dispatch("stream", gradio.props.value);
								},
								ondrag: (detail) => set(dragging, detail, true),
								get root() {
									return gradio.shared.root;
								},

								get sources() {
									return gradio.props.sources;
								},

								get active_source() {
									return get($1);
								},

								get pending() {
									return gradio.shared.loading_status.pending;
								},

								get streaming() {
									return gradio.props.streaming;
								},

								get loop() {
									return gradio.props.loop;
								},

								get max_file_size() {
									return gradio.shared.max_file_size;
								},
								handle_reset_value,
								get editable() {
									return gradio.props.editable;
								},
								onedit: () => gradio.dispatch("edit"),
								onplay: () => gradio.dispatch("play"),
								onpause: () => gradio.dispatch("pause"),
								onstop: () => gradio.dispatch("stop"),
								onstart_recording: () => gradio.dispatch("start_recording"),
								onpause_recording: () => gradio.dispatch("pause_recording"),
								onstop_recording: () => {
									gradio.dispatch("stop_recording");
									gradio.dispatch("input");
								},

								onupload: () => {
									gradio.dispatch("upload");
									gradio.dispatch("input");
								},

								onclear: () => {
									gradio.dispatch("clear");
									gradio.dispatch("input");
								},
								onerror: handle_error,
								onclose_stream: () => gradio.dispatch("close_stream", "stream"),
								get i18n() {
									return gradio.i18n;
								},

								get waveform_settings() {
									return waveform_settings;
								},

								get waveform_options() {
									return gradio.props.waveform_options;
								},

								get trim_region_settings() {
									return trim_region_settings;
								},

								get stream_every() {
									return gradio.props.stream_every;
								},

								get stream_state() {
									return gradio.shared.loading_status.stream_state;
								},
								upload: (...args) => gradio.shared.client.upload(...args),
								stream_handler: (...args) => gradio.shared.client.stream(...args),
								get time_limit() {
									return gradio.shared.loading_status.time_limit;
								},

								get upload_promise() {
									return get(upload_promise);
								},

								set upload_promise($$value) {
									set(upload_promise, $$value, true);
								},

								get initial_value() {
									return get(initial_value);
								},

								set initial_value($$value) {
									set(initial_value, $$value, true);
								},

								get recording() {
									return get(recording);
								},

								set recording($$value) {
									set(recording, $$value, true);
								},

								get dragging() {
									return get(dragging);
								},

								set dragging($$value) {
									set(dragging, $$value, true);
								},

								get playback_position() {
									return gradio.props.playback_position;
								},

								set playback_position($$value) {
									gradio.props.playback_position = $$value;
								},

								children: ($$anchor, $$slotProps) => {
									UploadText($$anchor, {
										get i18n() {
											return gradio.i18n;
										},
										type: 'audio'
									});
								},
								$$slots: { default: true }
							});
						}

						append($$anchor, fragment_4);
					},
					$$slots: { default: true }
				});
			}
		};

		if_block(node, ($$render) => {
			if (!gradio.shared.interactive) $$render(consequent); else $$render(alternate, false);
		});
	}

	append($$anchor, fragment);
	pop();
}

export { InteractiveAudio as BaseInteractiveAudio, AudioPlayer as BasePlayer, StaticAudio as BaseStaticAudio, Index as default };
//# sourceMappingURL=index-D65tv47j.js.map
