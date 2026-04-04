import { p as prop, i as if_block, k as each, a as set_attribute, u as index, f as set_style, b as set_class, t as remove_input_defaults, d as bind_this, F as set_value } from './i18n-CMo5lpzy.js';
import { aj as delegate, R as push, u as state, v as proxy, y as user_effect, w as get, aa as onMount, ae as onDestroy, V as child, Y as reset, a as append, T as pop, a5 as user_derived, x as set, X as sibling, t as template_effect, Z as event, W as from_html, S as first_child, a0 as set_text, a6 as comment, au as prepare_files, M as tick } from './index-BvBk1Iap.js';
import { k as key } from './key-BV46G5Fv.js';
import { I as IconButton, V as Video$1 } from './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';
import { B as BlockLabel } from './BlockLabel-oGSqUii_.js';
import { D as DownloadLink } from './DownloadLink-KvrFKyMn.js';
import { E as Empty } from './Empty-CHa-gxv7.js';
import { S as ShareButton } from './ShareButton-Dz0y1C1K.js';
import { D as Download } from './Download-pFhUaTl4.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-Ba0GidgU.js';
import { f as format_time, u as uploadToHuggingFace } from './utils.svelte-CYLB_jIA.js';
import { M as Maximize } from './Maximize-B_43ylTs.js';
import { T as Trim, V as VolumeLevels, P as Pause } from './VolumeLevels-DHAc0pUL.js';
import { P as Play } from './Play-CWcFSzc7.js';
import { U as Undo } from './Undo-BxcJDM8E.js';
import { b as loadFfmpeg, t as trimVideo, V as Video } from './Video-B3xDM6GN.js';
import './Upload-CPeZiqam.js';
import { M as ModifyUpload } from './ModifyUpload-B5K-5Hqx.js';

var root_1$2 = from_html(`<div class="load-wrap svelte-mctcyk"><span aria-label="loading timeline" class="loader svelte-mctcyk"></span></div>`);
var root_3$2 = from_html(`<img draggable="false" class="svelte-mctcyk"/>`);
var root_2$2 = from_html(`<div id="timeline" class="thumbnail-wrapper svelte-mctcyk"><button aria-label="start drag handle for trimming video" class="handle left svelte-mctcyk"></button> <div class="opaque-layer svelte-mctcyk"></div> <!> <button aria-label="end drag handle for trimming video" class="handle right svelte-mctcyk"></button></div>`);
var root$3 = from_html(`<div class="container svelte-mctcyk"><!></div>`);

function VideoTimeline($$anchor, $$props) {
	push($$props, true);

	let videoElement = prop($$props, 'videoElement', 7),
		trimmedDuration = prop($$props, 'trimmedDuration', 15, null),
		dragStart = prop($$props, 'dragStart', 15, 0),
		dragEnd = prop($$props, 'dragEnd', 15, 0),
		loadingTimeline = prop($$props, 'loadingTimeline', 15, false);

	let thumbnails = state(proxy([]));
	let numberOfThumbnails = 10;
	let videoDuration;
	let leftHandlePosition = state(0);
	let rightHandlePosition = state(100);
	let dragging = state(null);

	const startDragging = (side) => {
		set(dragging, side, true);
	};

	let loadingTimelineValue = user_derived(() => get(thumbnails).length !== numberOfThumbnails);

	user_effect(() => {
		loadingTimeline(get(loadingTimelineValue));
	});

	const stopDragging = () => {
		set(dragging, null);
	};

	const drag = (event, distance) => {
		if (get(dragging)) {
			const timeline = document.getElementById("timeline");

			if (!timeline) return;

			const rect = timeline.getBoundingClientRect();
			let newPercentage = (event.clientX - rect.left) / rect.width * 100;

			if (distance) {
				// Move handle based on arrow key press
				newPercentage = get(dragging) === "left"
					? get(leftHandlePosition) + distance
					: get(rightHandlePosition) + distance;
			} else {
				// Move handle based on mouse drag
				newPercentage = (event.clientX - rect.left) / rect.width * 100;
			}

			newPercentage = Math.max(0, Math.min(newPercentage, 100)); // Keep within 0 and 100

			if (get(dragging) === "left") {
				set(leftHandlePosition, Math.min(newPercentage, get(rightHandlePosition)), true);

				// Calculate the new time and set it for the videoElement
				const newTimeLeft = get(leftHandlePosition) / 100 * videoDuration;

				videoElement().currentTime = newTimeLeft;
				dragStart(newTimeLeft);
			} else if (get(dragging) === "right") {
				set(rightHandlePosition, Math.max(newPercentage, get(leftHandlePosition)), true);

				const newTimeRight = get(rightHandlePosition) / 100 * videoDuration;

				videoElement().currentTime = newTimeRight;
				dragEnd(newTimeRight);
			}

			const startTime = get(leftHandlePosition) / 100 * videoDuration;
			const endTime = get(rightHandlePosition) / 100 * videoDuration;

			trimmedDuration(endTime - startTime);
			set(leftHandlePosition, get(leftHandlePosition), true);
			set(rightHandlePosition, get(rightHandlePosition), true);
		}
	};

	const moveHandle = (e) => {
		if (get(dragging)) {
			// Calculate the movement distance as a percentage of the video duration
			const distance = 1 / videoDuration * 100;

			if (e.key === "ArrowLeft") {
				drag({ clientX: 0 }, -distance);
			} else if (e.key === "ArrowRight") {
				drag({ clientX: 0 }, distance);
			}
		}
	};

	const generateThumbnail = () => {
		const canvas = document.createElement("canvas");
		const ctx = canvas.getContext("2d");

		if (!ctx) return;

		canvas.width = videoElement().videoWidth;
		canvas.height = videoElement().videoHeight;
		ctx.drawImage(videoElement(), 0, 0, canvas.width, canvas.height);

		const thumbnail = canvas.toDataURL("image/jpeg", 0.7);

		set(thumbnails, [...get(thumbnails), thumbnail], true);
	};

	onMount(() => {
		const loadMetadata = () => {
			videoDuration = videoElement().duration;

			const interval = videoDuration / numberOfThumbnails;
			let captures = 0;

			const onSeeked = () => {
				generateThumbnail();
				captures++;

				if (captures < numberOfThumbnails) {
					videoElement().currentTime += interval;
				} else {
					videoElement().removeEventListener("seeked", onSeeked);
				}
			};

			videoElement().addEventListener("seeked", onSeeked);
			videoElement().currentTime = 0;
		};

		if (videoElement().readyState >= 1) {
			loadMetadata();
		} else {
			videoElement().addEventListener("loadedmetadata", loadMetadata);
		}
	});

	onDestroy(() => {
		window.removeEventListener("mousemove", drag);
		window.removeEventListener("mouseup", stopDragging);
		window.removeEventListener("keydown", moveHandle);
	});

	onMount(() => {
		window.addEventListener("mousemove", drag);
		window.addEventListener("mouseup", stopDragging);
		window.addEventListener("keydown", moveHandle);
	});

	var div = root$3();
	var node = child(div);

	{
		var consequent = ($$anchor) => {
			var div_1 = root_1$2();

			append($$anchor, div_1);
		};

		var alternate = ($$anchor) => {
			var div_2 = root_2$2();
			var button = child(div_2);

			button.__mousedown = () => startDragging("left");

			button.__keydown = (e) => {
				if (e.key === "ArrowLeft" || e.key == "ArrowRight") {
					startDragging("left");
				}
			};

			var div_3 = sibling(button, 2);
			var node_1 = sibling(div_3, 2);

			each(node_1, 17, () => get(thumbnails), index, ($$anchor, thumbnail, i) => {
				var img = root_3$2();

				set_attribute(img, 'alt', `frame-${i}`);
				template_effect(() => set_attribute(img, 'src', get(thumbnail)));
				append($$anchor, img);
			});

			var button_1 = sibling(node_1, 2);

			button_1.__mousedown = () => startDragging("right");

			button_1.__keydown = (e) => {
				if (e.key === "ArrowLeft" || e.key == "ArrowRight") {
					startDragging("right");
				}
			};

			reset(div_2);

			template_effect(() => {
				set_style(button, `left: ${get(leftHandlePosition) ?? ''}%;`);
				set_style(div_3, `left: ${get(leftHandlePosition) ?? ''}%; right: ${100 - get(rightHandlePosition)}%`);
				set_style(button_1, `left: ${get(rightHandlePosition) ?? ''}%;`);
			});

			event('blur', button, stopDragging);
			event('blur', button_1, stopDragging);
			append($$anchor, div_2);
		};

		if_block(node, ($$render) => {
			if (get(loadingTimelineValue)) $$render(consequent); else $$render(alternate, false);
		});
	}

	reset(div);
	append($$anchor, div);
	pop();
}

delegate(['mousedown', 'keydown']);

var root_1$1 = from_html(`<div class="timeline-wrapper svelte-1orxdv"><!></div>`);
var root_2$1 = from_html(`<time aria-label="duration of selected region in seconds"> </time> <div class="edit-buttons svelte-1orxdv"><button>Trim</button> <button>Cancel</button></div>`, 1);
var root_3$1 = from_html(`<div class="svelte-1orxdv"></div>`);
var root_4 = from_html(`<!> <!>`, 1);
var root$2 = from_html(`<div><!> <div class="controls svelte-1orxdv" data-testid="waveform-controls"><!></div></div> <!>`, 1);

function VideoControls($$anchor, $$props) {
	push($$props, true);

	let videoElement = prop($$props, 'videoElement', 3, undefined),
		showRedo = prop($$props, 'showRedo', 3, false),
		interactive = prop($$props, 'interactive', 3, true),
		mode = prop($$props, 'mode', 15, ""),
		processingVideo = prop($$props, 'processingVideo', 15, false),
		value = prop($$props, 'value', 3, null),
		show_download_button = prop($$props, 'show_download_button', 3, false),
		handle_clear = prop($$props, 'handle_clear', 3, () => {}),
		has_change_history = prop($$props, 'has_change_history', 3, false);

	let ffmpeg;

	onMount(async () => {
		ffmpeg = await loadFfmpeg();
	});

	user_effect(() => {
		if (mode() === "edit" && get(trimmedDuration) === null && videoElement()) {
			set(trimmedDuration, videoElement().duration, true);
		}
	});

	let trimmedDuration = state(null);
	let dragStart = state(0);
	let dragEnd = state(0);
	let loadingTimeline = state(false);

	const toggleTrimmingMode = () => {
		if (mode() === "edit") {
			mode("");

			if (videoElement()) {
				set(trimmedDuration, videoElement().duration, true);
			}
		} else {
			mode("edit");
		}
	};

	var fragment = root$2();
	var div = first_child(fragment);
	let classes;
	var node = child(div);

	{
		var consequent = ($$anchor) => {
			var div_1 = root_1$1();
			var node_1 = child(div_1);

			VideoTimeline(node_1, {
				get videoElement() {
					return videoElement();
				},

				get dragStart() {
					return get(dragStart);
				},

				set dragStart($$value) {
					set(dragStart, $$value, true);
				},

				get dragEnd() {
					return get(dragEnd);
				},

				set dragEnd($$value) {
					set(dragEnd, $$value, true);
				},

				get trimmedDuration() {
					return get(trimmedDuration);
				},

				set trimmedDuration($$value) {
					set(trimmedDuration, $$value, true);
				},

				get loadingTimeline() {
					return get(loadingTimeline);
				},

				set loadingTimeline($$value) {
					set(loadingTimeline, $$value, true);
				}
			});

			reset(div_1);
			append($$anchor, div_1);
		};

		if_block(node, ($$render) => {
			if (mode() === "edit" && videoElement()) $$render(consequent);
		});
	}

	var div_2 = sibling(node, 2);
	var node_2 = child(div_2);

	{
		var consequent_1 = ($$anchor) => {
			var fragment_1 = root_2$1();
			var time = first_child(fragment_1);
			let classes_1;
			var text = child(time, true);

			reset(time);

			var div_3 = sibling(time, 2);
			var button = child(div_3);
			let classes_2;

			button.__click = () => {
				if (!videoElement()) return;

				mode("");
				processingVideo(true);

				trimVideo(ffmpeg, get(dragStart), get(dragEnd), videoElement()).then((videoBlob) => {
					$$props.handle_trim_video(videoBlob);
				}).then(() => {
					processingVideo(false);
				});
			};

			var button_1 = sibling(button, 2);
			let classes_3;

			button_1.__click = toggleTrimmingMode;
			reset(div_3);

			template_effect(
				($0) => {
					classes_1 = set_class(time, 1, 'svelte-1orxdv', null, classes_1, { hidden: get(loadingTimeline) });
					set_text(text, $0);
					classes_2 = set_class(button, 1, 'text-button svelte-1orxdv', null, classes_2, { hidden: get(loadingTimeline) });
					classes_3 = set_class(button_1, 1, 'text-button svelte-1orxdv', null, classes_3, { hidden: get(loadingTimeline) });
				},
				[() => format_time(get(trimmedDuration))]
			);

			append($$anchor, fragment_1);
		};

		var alternate = ($$anchor) => {
			var div_4 = root_3$1();

			append($$anchor, div_4);
		};

		if_block(node_2, ($$render) => {
			if (mode() === "edit" && get(trimmedDuration) !== null) $$render(consequent_1); else $$render(alternate, false);
		});
	}

	reset(div_2);
	reset(div);

	var node_3 = sibling(div, 2);

	{
		let $0 = user_derived(() => show_download_button() ? value()?.url : null);

		ModifyUpload(node_3, {
			get i18n() {
				return $$props.i18n;
			},
			onclear: () => handle_clear()(),
			get download() {
				return get($0);
			},

			children: ($$anchor, $$slotProps) => {
				var fragment_2 = root_4();
				var node_4 = first_child(fragment_2);

				{
					var consequent_2 = ($$anchor) => {
						{
							let $0 = user_derived(() => processingVideo() || !has_change_history());

							IconButton($$anchor, {
								get Icon() {
									return Undo;
								},
								label: 'Reset video to initial value',
								get disabled() {
									return get($0);
								},

								onclick: () => {
									$$props.handle_reset_value();
									mode("");
								}
							});
						}
					};

					if_block(node_4, ($$render) => {
						if (showRedo() && mode() === "") $$render(consequent_2);
					});
				}

				var node_5 = sibling(node_4, 2);

				{
					var consequent_3 = ($$anchor) => {
						IconButton($$anchor, {
							get Icon() {
								return Trim;
							},
							label: 'Trim video to selection',
							get disabled() {
								return processingVideo();
							},
							onclick: toggleTrimmingMode
						});
					};

					if_block(node_5, ($$render) => {
						if (interactive() && mode() === "") $$render(consequent_3);
					});
				}

				append($$anchor, fragment_2);
			},
			$$slots: { default: true }
		});
	}

	template_effect(() => classes = set_class(div, 1, 'container svelte-1orxdv', null, classes, { hidden: mode() !== "edit" }));
	append($$anchor, fragment);
	pop();
}

delegate(['click']);

var root$1 = from_html(`<input id="volume" class="volume-slider svelte-h4x5cd" type="range" min="0" max="1" step="0.01"/>`);

function VolumeControl($$anchor, $$props) {
	push($$props, true);

	let current_volume = prop($$props, 'current_volume', 15, 1),
		show_volume_slider = prop($$props, 'show_volume_slider', 15, false);

	let volume_element = state(void 0);

	onMount(() => {
		adjustSlider();
	});

	const adjustSlider = () => {
		let slider = get(volume_element);

		if (!slider) return;

		slider.style.background = `linear-gradient(to right, white ${current_volume() * 100}%, rgba(255, 255, 255, 0.3) ${current_volume() * 100}%)`;
	};

	user_effect(() => {
		current_volume();
		adjustSlider();
	});

	var input = root$1();

	remove_input_defaults(input);
	input.__focusout = () => show_volume_slider(false);

	input.__input = (e) => {
		if (e.target instanceof HTMLInputElement) {
			current_volume(parseFloat(e.target.value));
		}
	};

	bind_this(input, ($$value) => set(volume_element, $$value), () => get(volume_element));
	template_effect(() => set_value(input, current_volume()));
	append($$anchor, input);
	pop();
}

delegate(['focusout', 'input']);

var root_2 = from_html(`<track kind="captions" default/>`);
var root_8 = from_html(`<div role="button" tabindex="0" class="icon svelte-1k28h7x" aria-label="full-screen"><!></div>`);
var root_1 = from_html(`<div class="wrap svelte-1k28h7x"><div><!></div> <div class="controls svelte-1k28h7x"><div class="inner svelte-1k28h7x"><span role="button" tabindex="0" class="icon svelte-1k28h7x" aria-label="play-pause-replay-button"><!></span> <span class="time svelte-1k28h7x"> </span>  <progress class="svelte-1k28h7x"></progress> <div class="volume-control-wrapper svelte-1k28h7x"><button class="icon volume-button svelte-1k28h7x" aria-label="Adjust volume"><!></button> <!></div> <!></div></div></div> <!>`, 1);

function Player($$anchor, $$props) {
	push($$props, true);

	let root = prop($$props, 'root', 3, ""),
		subtitle = prop($$props, 'subtitle', 3, null),
		label = prop($$props, 'label', 3, "test"),
		interactive = prop($$props, 'interactive', 3, false),
		handle_change = prop($$props, 'handle_change', 3, () => {}),
		handle_reset_value = prop($$props, 'handle_reset_value', 3, () => {}),
		is_stream = prop($$props, 'is_stream', 3, undefined),
		show_download_button = prop($$props, 'show_download_button', 3, false),
		value = prop($$props, 'value', 3, null),
		handle_clear = prop($$props, 'handle_clear', 3, () => {}),
		has_change_history = prop($$props, 'has_change_history', 3, false),
		playback_position = prop($$props, 'playback_position', 15);

	let time = state(0);
	let duration = state(0);
	let paused = state(true);
	let video = state(void 0);
	let processingVideo = state(false);
	let show_volume_slider = state(false);
	let current_volume = state(1);
	let is_fullscreen = state(false);

	function handleMove(e) {
		if (!get(duration)) return;

		if (e.type === "click") {
			handle_click(e);

			return;
		}

		if (e.type !== "touchmove" && !(e.buttons & 1)) return;

		const clientX = e.type === "touchmove" ? e.touches[0].clientX : e.clientX;
		const { left, right } = e.currentTarget.getBoundingClientRect();

		set(time, get(duration) * (clientX - left) / (right - left));
	}

	async function play_pause() {
		if (!get(video)) return;

		if (document.fullscreenElement != get(video)) {
			const isPlaying = get(video).currentTime > 0 && !get(video).paused && !get(video).ended && get(video).readyState > get(video).HAVE_CURRENT_DATA;

			if (!isPlaying) {
				await get(video).play();
			} else get(video).pause();
		}
	}

	function handle_click(e) {
		if (!get(duration)) return;

		const { left, right } = e.currentTarget.getBoundingClientRect();

		set(time, get(duration) * (e.clientX - left) / (right - left));
	}

	function handle_end() {
		$$props.onstop?.();
		$$props.onend?.();
	}

	const handle_trim_video = async (videoBlob) => {
		let _video_blob = new File([videoBlob], "video.mp4");
		const val = await prepare_files([_video_blob]);
		let value = (await $$props.upload(val, root()))?.filter(Boolean)[0];

		handle_change()(value);
	};

	function open_full_screen() {
		if (!get(video)) return;

		if (!get(is_fullscreen)) {
			get(video).requestFullscreen();
		} else {
			document.exitFullscreen();
		}
	}

	function handleFullscreenChange() {
		set(is_fullscreen, document.fullscreenElement === get(video));

		if (get(video)) {
			get(video).controls = get(is_fullscreen);
		}
	}

	let last_synced_volume = 1;
	let previous_video;

	// Tolerance for floating-point comparison of volume values
	const VOLUME_EPSILON = 0.001;

	function handleVolumeChange() {
		if (get(video) && Math.abs(get(video).volume - last_synced_volume) > VOLUME_EPSILON) {
			set(current_volume, get(video).volume, true);
			last_synced_volume = get(video).volume;
		}
	}

	onMount(() => {
		document.addEventListener("fullscreenchange", handleFullscreenChange);

		return () => {
			document.removeEventListener("fullscreenchange", handleFullscreenChange);
		};
	});

	onDestroy(() => {
		if (get(video)) {
			get(video).removeEventListener("volumechange", handleVolumeChange);
		}
	});

	user_effect(() => {
		if (get(video) && get(video) !== previous_video) {
			if (previous_video) {
				previous_video.removeEventListener("volumechange", handleVolumeChange);
			}

			get(video).addEventListener("volumechange", handleVolumeChange);
			previous_video = get(video);
		}
	});

	user_effect(() => {
		playback_position(get(time));
	});

	user_effect(() => {
		if (playback_position() !== get(time) && get(video)) {
			get(video).currentTime = playback_position();
		}
	});

	user_effect(() => {
		if (get(video) && !get(is_fullscreen)) {
			if (Math.abs(get(video).volume - get(current_volume)) > VOLUME_EPSILON) {
				get(video).volume = get(current_volume);
				last_synced_volume = get(current_volume);
			}

			get(video).controls = false;
		}
	});

	user_effect(() => {
		if (get(video) && get(is_fullscreen)) {
			last_synced_volume = get(video).volume;
		}
	});

	var fragment = root_1();
	var div = first_child(fragment);
	var div_1 = child(div);
	let classes;
	var node = child(div_1);

	{
		let $0 = user_derived(() => `${label()}-player`);

		Video(node, {
			get src() {
				return $$props.src;
			},
			preload: 'auto',
			get autoplay() {
				return $$props.autoplay;
			},

			get loop() {
				return $$props.loop;
			},

			get is_stream() {
				return is_stream();
			},

			get controls() {
				return get(is_fullscreen);
			},
			onclick: play_pause,
			onplay: () => $$props.onplay?.(),
			onpause: () => $$props.onpause?.(),
			onerror: (error) => $$props.onerror?.(error),
			onended: handle_end,
			get 'data-testid'() {
				return get($0);
			},

			get processingVideo() {
				return get(processingVideo);
			},
			onloadstart: () => $$props.onloadstart?.(),
			onloadeddata: () => $$props.onloadeddata?.(),
			onloadedmetadata: () => $$props.onloadedmetadata?.(),
			get currentTime() {
				return get(time);
			},

			set currentTime($$value) {
				set(time, $$value, true);
			},

			get duration() {
				return get(duration);
			},

			set duration($$value) {
				set(duration, $$value, true);
			},

			get paused() {
				return get(paused);
			},

			set paused($$value) {
				set(paused, $$value, true);
			},

			get node() {
				return get(video);
			},

			set node($$value) {
				set(video, $$value, true);
			},

			children: ($$anchor, $$slotProps) => {
				var track = root_2();

				template_effect(() => set_attribute(track, 'src', subtitle()));
				append($$anchor, track);
			},
			$$slots: { default: true }
		});
	}

	reset(div_1);

	var div_2 = sibling(div_1, 2);
	var div_3 = child(div_2);
	var span = child(div_3);

	span.__click = play_pause;
	span.__keydown = play_pause;

	var node_1 = child(span);

	{
		var consequent = ($$anchor) => {
			Undo($$anchor);
		};

		var alternate_1 = ($$anchor) => {
			var fragment_2 = comment();
			var node_2 = first_child(fragment_2);

			{
				var consequent_1 = ($$anchor) => {
					Play($$anchor);
				};

				var alternate = ($$anchor) => {
					Pause($$anchor);
				};

				if_block(
					node_2,
					($$render) => {
						if (get(paused)) $$render(consequent_1); else $$render(alternate, false);
					},
					true
				);
			}

			append($$anchor, fragment_2);
		};

		if_block(node_1, ($$render) => {
			if (get(time) === get(duration)) $$render(consequent); else $$render(alternate_1, false);
		});
	}

	reset(span);

	var span_1 = sibling(span, 2);
	var text = child(span_1);

	reset(span_1);

	var progress = sibling(span_1, 2);

	progress.__mousemove = handleMove;

	progress.__touchmove = (e) => {
		e.preventDefault();
		handleMove(e);
	};

	progress.__click = (e) => {
		e.stopPropagation();
		e.preventDefault();
		handle_click(e);
	};

	var div_4 = sibling(progress, 2);
	var button = child(div_4);

	button.__click = () => set(show_volume_slider, !get(show_volume_slider));

	let styles;
	var node_3 = child(button);

	VolumeLevels(node_3, {
		get currentVolume() {
			return get(current_volume);
		}
	});

	reset(button);

	var node_4 = sibling(button, 2);

	{
		var consequent_2 = ($$anchor) => {
			VolumeControl($$anchor, {
				get current_volume() {
					return get(current_volume);
				},

				set current_volume($$value) {
					set(current_volume, $$value, true);
				},

				get show_volume_slider() {
					return get(show_volume_slider);
				},

				set show_volume_slider($$value) {
					set(show_volume_slider, $$value, true);
				}
			});
		};

		if_block(node_4, ($$render) => {
			if (get(show_volume_slider)) $$render(consequent_2);
		});
	}

	reset(div_4);

	var node_5 = sibling(div_4, 2);

	{
		var consequent_3 = ($$anchor) => {
			var div_5 = root_8();

			div_5.__click = open_full_screen;

			var node_6 = child(div_5);

			Maximize(node_6);
			reset(div_5);
			event('keypress', div_5, open_full_screen);
			append($$anchor, div_5);
		};

		if_block(node_5, ($$render) => {
			if (!get(show_volume_slider)) $$render(consequent_3);
		});
	}

	reset(div_3);
	reset(div_2);
	reset(div);

	var node_7 = sibling(div, 2);

	{
		var consequent_4 = ($$anchor) => {
			VideoControls($$anchor, {
				get videoElement() {
					return get(video);
				},
				showRedo: true,
				handle_trim_video,
				get handle_reset_value() {
					return handle_reset_value();
				},

				get value() {
					return value();
				},

				get i18n() {
					return $$props.i18n;
				},

				get show_download_button() {
					return show_download_button();
				},

				get handle_clear() {
					return handle_clear();
				},

				get has_change_history() {
					return has_change_history();
				},

				get processingVideo() {
					return get(processingVideo);
				},

				set processingVideo($$value) {
					set(processingVideo, $$value, true);
				}
			});
		};

		if_block(node_7, ($$render) => {
			if (interactive()) $$render(consequent_4);
		});
	}

	template_effect(
		($0, $1) => {
			classes = set_class(div_1, 1, 'mirror-wrap svelte-1k28h7x', null, classes, { mirror: $$props.mirror });
			set_text(text, `${$0 ?? ''} / ${$1 ?? ''}`);
			set_value(progress, get(time) / get(duration) || 0);

			styles = set_style(button, '', styles, {
				color: get(show_volume_slider) ? "var(--color-accent)" : "white"
			});
		},
		[
			() => format_time(get(time)),
			() => format_time(get(duration))
		]
	);

	append($$anchor, fragment);
	pop();
}

delegate(['click', 'keydown', 'mousemove', 'touchmove']);

var root_5 = from_html(`<!> <!>`, 1);
var root_3 = from_html(`<!> <div data-testid="download-div"><!></div>`, 1);
var root = from_html(`<!> <!>`, 1);

function VideoPreview($$anchor, $$props) {
	push($$props, true);

	let value = prop($$props, 'value', 15, null),
		subtitle = prop($$props, 'subtitle', 3, null),
		label = prop($$props, 'label', 3, undefined),
		show_label = prop($$props, 'show_label', 3, true),
		buttons = prop($$props, 'buttons', 3, null),
		on_custom_button_click = prop($$props, 'on_custom_button_click', 3, null),
		display_icon_button_wrapper_top_corner = prop($$props, 'display_icon_button_wrapper_top_corner', 3, false),
		playback_position = prop($$props, 'playback_position', 15);

	let old_value = state(null);
	let old_subtitle = state(null);

	user_effect(() => {
		if (value()) {
			$$props.onchange?.(value());
		}
	});

	user_effect(() => {
		async function updateValue() {
			// needed to bust subtitle caching issues on Chrome
			if (value() !== get(old_value) && subtitle() !== get(old_subtitle) && get(old_subtitle) !== null) {
				set(old_value, value(), true);
				value(null);
				await tick();
				value(get(old_value));
			}

			set(old_value, value(), true);
			set(old_subtitle, subtitle());
		}

		updateValue();
	});

	var fragment = root();
	var node = first_child(fragment);

	{
		let $0 = user_derived(() => label() || "Video");

		BlockLabel(node, {
			get show_label() {
				return show_label();
			},

			get Icon() {
				return Video$1;
			},

			get label() {
				return get($0);
			}
		});
	}

	var node_1 = sibling(node, 2);

	{
		var consequent = ($$anchor) => {
			Empty($$anchor, {
				unpadded_box: true,
				size: 'large',
				children: ($$anchor, $$slotProps) => {
					Video$1($$anchor);
				},
				$$slots: { default: true }
			});
		};

		var alternate = ($$anchor) => {
			var fragment_3 = root_3();
			var node_2 = first_child(fragment_3);

			key(node_2, () => value().url, ($$anchor) => {
				{
					let $0 = user_derived(() => subtitle()?.url);

					Player($$anchor, {
						get src() {
							return value().url;
						},

						get subtitle() {
							return get($0);
						},

						get is_stream() {
							return value().is_stream;
						},

						get autoplay() {
							return $$props.autoplay;
						},
						onplay: () => $$props.onplay?.(),
						onpause: () => $$props.onpause?.(),
						onstop: () => $$props.onstop?.(),
						onend: () => $$props.onend?.(),
						onloadedmetadata: () => {
							// Deal with `<video>`'s `loadedmetadata` event as `VideoPreview`'s `load` event
							// to represent not only the video is loaded but also the metadata is loaded
							// so its dimensions (w/h) are known. This is used for Chatbot's auto scroll.
							$$props.onload?.();
						},
						mirror: false,
						get label() {
							return label();
						},

						get loop() {
							return $$props.loop;
						},
						interactive: false,
						get upload() {
							return $$props.upload;
						},

						get i18n() {
							return $$props.i18n;
						},

						get playback_position() {
							return playback_position();
						},

						set playback_position($$value) {
							playback_position($$value);
						}
					});
				}
			});

			var div = sibling(node_2, 2);
			var node_3 = child(div);

			{
				let $0 = user_derived(() => buttons() ?? ["download", "share"]);

				IconButtonWrapper(node_3, {
					get display_top_corner() {
						return display_icon_button_wrapper_top_corner();
					},

					get buttons() {
						return get($0);
					},

					get on_custom_button_click() {
						return on_custom_button_click();
					},

					children: ($$anchor, $$slotProps) => {
						var fragment_5 = root_5();
						var node_4 = first_child(fragment_5);

						{
							var consequent_1 = ($$anchor) => {
								{
									let $0 = user_derived(() => value().is_stream
										? value().url?.replace("playlist.m3u8", "playlist-file")
										: value().url);

									let $1 = user_derived(() => value().orig_name || value().path);

									DownloadLink($$anchor, {
										get href() {
											return get($0);
										},

										get download() {
											return get($1);
										},

										children: ($$anchor, $$slotProps) => {
											IconButton($$anchor, {
												get Icon() {
													return Download;
												},
												label: 'Download'
											});
										},
										$$slots: { default: true }
									});
								}
							};

							if_block(node_4, ($$render) => {
								if (buttons()?.some((btn) => typeof btn === "string" && btn === "download")) $$render(consequent_1);
							});
						}

						var node_5 = sibling(node_4, 2);

						{
							var consequent_2 = ($$anchor) => {
								ShareButton($$anchor, {
									get i18n() {
										return $$props.i18n;
									},
									onerror: (detail) => $$props.onerror?.(detail),
									onshare: (detail) => $$props.onshare?.(detail),
									get value() {
										return value();
									},

									formatter: async (value) => {
										if (!value) return "";

										let url = await uploadToHuggingFace(value.data);

										return url;
									}
								});
							};

							if_block(node_5, ($$render) => {
								if (buttons()?.some((btn) => typeof btn === "string" && btn === "share")) $$render(consequent_2);
							});
						}

						append($$anchor, fragment_5);
					},
					$$slots: { default: true }
				});
			}

			reset(div);
			append($$anchor, fragment_3);
		};

		if_block(node_1, ($$render) => {
			if (!value() || value().url === undefined) $$render(consequent); else $$render(alternate, false);
		});
	}

	append($$anchor, fragment);
	pop();
}

const VideoPreview$1 = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
	__proto__: null,
	default: VideoPreview
}, Symbol.toStringTag, { value: 'Module' }));

export { Player as P, VideoPreview as V, VideoPreview$1 as a };
//# sourceMappingURL=VideoPreview-B9h4wUga.js.map
