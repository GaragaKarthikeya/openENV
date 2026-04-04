import './async-D55cHugf.js';
import { d as bind_props, f as attr_class, a as attr, g as attr_style, i as stringify, e as ensure_array_like } from './index-u8mz_F03.js';
import { o as onDestroy } from './index-server-CQz6EZl_.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { D as DownloadLink } from './DownloadLink-BMNxSkQ8.js';
import { I as IconButton } from './IconButton-D82v5nCM.js';
import { E as Empty } from './Empty-BHvMDNv5.js';
import { S as ShareButton } from './ShareButton-DnAPtcgb.js';
import { D as Download } from './Download-ByiErn53.js';
import { V as Video$1 } from './Video-_1zl9-Cr.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { a as prepare_files } from './2-BAeILJ1g.js';
import { f as format_time, u as uploadToHuggingFace } from './utils.svelte-aYomt3KL.js';
import { M as Maximize } from './Maximize-B77VDSzq.js';
import { V as VolumeLevels, T as Trim, P as Pause } from './VolumeLevels-C0Xso12f.js';
import { P as Play } from './Play-BlZWjD1q.js';
import { U as Undo } from './Undo-Col2BcUY.js';
import { V as Video } from './Video2-De5bYQY6.js';
import { M as ModifyUpload } from './ModifyUpload-D_fhEqLO.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './Clear-DH-TDCgr.js';
import './Edit-W_0aHh0i.js';

const prettyBytes = (bytes) => {
  let units = ["B", "KB", "MB", "GB", "PB"];
  let i = 0;
  while (bytes > 1024) {
    bytes /= 1024;
    i++;
  }
  let unit = units[i];
  return bytes.toFixed(1) + " " + unit;
};
const playable = () => {
  return true;
};
function loaded(node, { autoplay }) {
  async function handle_playback() {
    if (!autoplay) return;
    await node.play();
  }
  node.addEventListener("loadeddata", handle_playback);
  return {
    destroy() {
      node.removeEventListener("loadeddata", handle_playback);
    }
  };
}
function VideoTimeline($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      videoElement,
      trimmedDuration = null,
      dragStart = 0,
      dragEnd = 0,
      loadingTimeline = false
    } = $$props;
    let thumbnails = [];
    let numberOfThumbnails = 10;
    let videoDuration;
    let leftHandlePosition = 0;
    let rightHandlePosition = 100;
    let dragging = null;
    let loadingTimelineValue = thumbnails.length !== numberOfThumbnails;
    const stopDragging = () => {
      dragging = null;
    };
    const drag = (event, distance) => {
      if (dragging) {
        const timeline = document.getElementById("timeline");
        if (!timeline) return;
        const rect = timeline.getBoundingClientRect();
        let newPercentage = (event.clientX - rect.left) / rect.width * 100;
        if (distance) {
          newPercentage = dragging === "left" ? leftHandlePosition + distance : rightHandlePosition + distance;
        } else {
          newPercentage = (event.clientX - rect.left) / rect.width * 100;
        }
        newPercentage = Math.max(0, Math.min(newPercentage, 100));
        if (dragging === "left") {
          leftHandlePosition = Math.min(newPercentage, rightHandlePosition);
          const newTimeLeft = leftHandlePosition / 100 * videoDuration;
          videoElement.currentTime = newTimeLeft;
          dragStart = newTimeLeft;
        } else if (dragging === "right") {
          rightHandlePosition = Math.max(newPercentage, leftHandlePosition);
          const newTimeRight = rightHandlePosition / 100 * videoDuration;
          videoElement.currentTime = newTimeRight;
          dragEnd = newTimeRight;
        }
        const startTime = leftHandlePosition / 100 * videoDuration;
        const endTime = rightHandlePosition / 100 * videoDuration;
        trimmedDuration = endTime - startTime;
        leftHandlePosition = leftHandlePosition;
        rightHandlePosition = rightHandlePosition;
      }
    };
    const moveHandle = (e) => {
      if (dragging) {
        const distance = 1 / videoDuration * 100;
        if (e.key === "ArrowLeft") {
          drag({ clientX: 0 }, -distance);
        } else if (e.key === "ArrowRight") {
          drag({ clientX: 0 }, distance);
        }
      }
    };
    onDestroy(() => {
      window.removeEventListener("mousemove", drag);
      window.removeEventListener("mouseup", stopDragging);
      window.removeEventListener("keydown", moveHandle);
    });
    $$renderer2.push(`<div class="container svelte-mctcyk">`);
    if (loadingTimelineValue) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="load-wrap svelte-mctcyk"><span aria-label="loading timeline" class="loader svelte-mctcyk"></span></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`<div id="timeline" class="thumbnail-wrapper svelte-mctcyk"><button aria-label="start drag handle for trimming video" class="handle left svelte-mctcyk"${attr_style(`left: ${stringify(leftHandlePosition)}%;`)}></button> <div class="opaque-layer svelte-mctcyk"${attr_style(`left: ${stringify(leftHandlePosition)}%; right: ${stringify(100 - rightHandlePosition)}%`)}></div> <!--[-->`);
      const each_array = ensure_array_like(thumbnails);
      for (let i = 0, $$length = each_array.length; i < $$length; i++) {
        let thumbnail = each_array[i];
        $$renderer2.push(`<img${attr("src", thumbnail)}${attr("alt", `frame-${i}`)} draggable="false" class="svelte-mctcyk"/>`);
      }
      $$renderer2.push(`<!--]--> <button aria-label="end drag handle for trimming video" class="handle right svelte-mctcyk"${attr_style(`left: ${stringify(rightHandlePosition)}%;`)}></button></div>`);
    }
    $$renderer2.push(`<!--]--></div>`);
    bind_props($$props, { trimmedDuration, dragStart, dragEnd, loadingTimeline });
  });
}
function VideoControls($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      videoElement = void 0,
      showRedo = false,
      interactive = true,
      mode = "",
      handle_reset_value,
      handle_trim_video,
      processingVideo = false,
      i18n,
      value = null,
      show_download_button = false,
      handle_clear = () => {
      },
      has_change_history = false
    } = $$props;
    let trimmedDuration = null;
    let dragStart = 0;
    let dragEnd = 0;
    let loadingTimeline = false;
    const toggleTrimmingMode = () => {
      if (mode === "edit") {
        mode = "";
        if (videoElement) {
          trimmedDuration = videoElement.duration;
        }
      } else {
        mode = "edit";
      }
    };
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div${attr_class("container svelte-1orxdv", void 0, { "hidden": mode !== "edit" })}>`);
      if (mode === "edit" && videoElement) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<div class="timeline-wrapper svelte-1orxdv">`);
        VideoTimeline($$renderer3, {
          videoElement,
          get dragStart() {
            return dragStart;
          },
          set dragStart($$value) {
            dragStart = $$value;
            $$settled = false;
          },
          get dragEnd() {
            return dragEnd;
          },
          set dragEnd($$value) {
            dragEnd = $$value;
            $$settled = false;
          },
          get trimmedDuration() {
            return trimmedDuration;
          },
          set trimmedDuration($$value) {
            trimmedDuration = $$value;
            $$settled = false;
          },
          get loadingTimeline() {
            return loadingTimeline;
          },
          set loadingTimeline($$value) {
            loadingTimeline = $$value;
            $$settled = false;
          }
        });
        $$renderer3.push(`<!----></div>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> <div class="controls svelte-1orxdv" data-testid="waveform-controls">`);
      if (mode === "edit" && trimmedDuration !== null) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<time aria-label="duration of selected region in seconds"${attr_class("svelte-1orxdv", void 0, { "hidden": loadingTimeline })}>${escape_html(format_time(trimmedDuration))}</time> <div class="edit-buttons svelte-1orxdv"><button${attr_class("text-button svelte-1orxdv", void 0, { "hidden": loadingTimeline })}>Trim</button> <button${attr_class("text-button svelte-1orxdv", void 0, { "hidden": loadingTimeline })}>Cancel</button></div>`);
      } else {
        $$renderer3.push("<!--[!-->");
        $$renderer3.push(`<div class="svelte-1orxdv"></div>`);
      }
      $$renderer3.push(`<!--]--></div></div> `);
      ModifyUpload($$renderer3, {
        i18n,
        onclear: () => handle_clear(),
        download: show_download_button ? value?.url : null,
        children: ($$renderer4) => {
          if (showRedo && mode === "") {
            $$renderer4.push("<!--[-->");
            IconButton($$renderer4, {
              Icon: Undo,
              label: "Reset video to initial value",
              disabled: processingVideo || !has_change_history,
              onclick: () => {
                handle_reset_value();
                mode = "";
              }
            });
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]--> `);
          if (interactive && mode === "") {
            $$renderer4.push("<!--[-->");
            IconButton($$renderer4, {
              Icon: Trim,
              label: "Trim video to selection",
              disabled: processingVideo,
              onclick: toggleTrimmingMode
            });
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]-->`);
        }
      });
      $$renderer3.push(`<!---->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { mode, processingVideo });
  });
}
function Player($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      root = "",
      src,
      subtitle = null,
      mirror,
      autoplay,
      loop,
      label = "test",
      interactive = false,
      handle_change = () => {
      },
      handle_reset_value = () => {
      },
      upload,
      is_stream = void 0,
      i18n,
      show_download_button = false,
      value = null,
      handle_clear = () => {
      },
      has_change_history = false,
      playback_position = void 0,
      onplay,
      onpause,
      onstop,
      onend,
      onerror,
      onloadstart,
      onloadeddata,
      onloadedmetadata
    } = $$props;
    let time = 0;
    let duration = 0;
    let paused = true;
    let video = void 0;
    let processingVideo = false;
    let current_volume = 1;
    let is_fullscreen = false;
    async function play_pause() {
      if (!video) return;
      if (document.fullscreenElement != video) {
        const isPlaying = video.currentTime > 0 && !video.paused && !video.ended && video.readyState > video.HAVE_CURRENT_DATA;
        if (!isPlaying) {
          await video.play();
        } else video.pause();
      }
    }
    function handle_end() {
      onstop?.();
      onend?.();
    }
    const handle_trim_video = async (videoBlob) => {
      let _video_blob = new File([videoBlob], "video.mp4");
      const val = await prepare_files([_video_blob]);
      let value2 = (await upload(val, root))?.filter(Boolean)[0];
      handle_change(value2);
    };
    let last_synced_volume = 1;
    const VOLUME_EPSILON = 1e-3;
    function handleVolumeChange() {
      if (video && Math.abs(video.volume - last_synced_volume) > VOLUME_EPSILON) {
        current_volume = video.volume;
        last_synced_volume = video.volume;
      }
    }
    onDestroy(() => {
      if (video) {
        video.removeEventListener("volumechange", handleVolumeChange);
      }
    });
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div class="wrap svelte-1k28h7x"><div${attr_class("mirror-wrap svelte-1k28h7x", void 0, { "mirror": mirror })}>`);
      Video($$renderer3, {
        src,
        preload: "auto",
        autoplay,
        loop,
        is_stream,
        controls: is_fullscreen,
        onclick: play_pause,
        onplay: () => onplay?.(),
        onpause: () => onpause?.(),
        onerror: (error) => onerror?.(error),
        onended: handle_end,
        "data-testid": `${label}-player`,
        processingVideo,
        onloadstart: () => onloadstart?.(),
        onloadeddata: () => onloadeddata?.(),
        onloadedmetadata: () => onloadedmetadata?.(),
        get currentTime() {
          return time;
        },
        set currentTime($$value) {
          time = $$value;
          $$settled = false;
        },
        get duration() {
          return duration;
        },
        set duration($$value) {
          duration = $$value;
          $$settled = false;
        },
        get paused() {
          return paused;
        },
        set paused($$value) {
          paused = $$value;
          $$settled = false;
        },
        get node() {
          return video;
        },
        set node($$value) {
          video = $$value;
          $$settled = false;
        },
        children: ($$renderer4) => {
          $$renderer4.push(`<track kind="captions"${attr("src", subtitle)} default/>`);
        },
        $$slots: { default: true }
      });
      $$renderer3.push(`<!----></div> <div class="controls svelte-1k28h7x"><div class="inner svelte-1k28h7x"><span role="button" tabindex="0" class="icon svelte-1k28h7x" aria-label="play-pause-replay-button">`);
      if (time === duration) {
        $$renderer3.push("<!--[-->");
        Undo($$renderer3);
      } else {
        $$renderer3.push("<!--[!-->");
        if (paused) {
          $$renderer3.push("<!--[-->");
          Play($$renderer3);
        } else {
          $$renderer3.push("<!--[!-->");
          Pause($$renderer3);
        }
        $$renderer3.push(`<!--]-->`);
      }
      $$renderer3.push(`<!--]--></span> <span class="time svelte-1k28h7x">${escape_html(format_time(time))} / ${escape_html(format_time(duration))}</span>  <progress${attr("value", time / duration || 0)} class="svelte-1k28h7x"></progress> <div class="volume-control-wrapper svelte-1k28h7x"><button class="icon volume-button svelte-1k28h7x" aria-label="Adjust volume"${attr_style("", { color: "white" })}>`);
      VolumeLevels($$renderer3, { currentVolume: current_volume });
      $$renderer3.push(`<!----></button> `);
      {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--></div> `);
      {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<div role="button" tabindex="0" class="icon svelte-1k28h7x" aria-label="full-screen">`);
        Maximize($$renderer3);
        $$renderer3.push(`<!----></div>`);
      }
      $$renderer3.push(`<!--]--></div></div></div> `);
      if (interactive) {
        $$renderer3.push("<!--[-->");
        VideoControls($$renderer3, {
          videoElement: video,
          showRedo: true,
          handle_trim_video,
          handle_reset_value,
          value,
          i18n,
          show_download_button,
          handle_clear,
          has_change_history,
          get processingVideo() {
            return processingVideo;
          },
          set processingVideo($$value) {
            processingVideo = $$value;
            $$settled = false;
          }
        });
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { playback_position });
  });
}
function VideoPreview($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value = null,
      subtitle = null,
      label = void 0,
      show_label = true,
      autoplay,
      buttons = null,
      on_custom_button_click = null,
      loop,
      i18n,
      upload,
      display_icon_button_wrapper_top_corner = false,
      playback_position = void 0,
      onplay,
      onpause,
      onend,
      onstop,
      onload,
      onchange,
      onerror,
      onshare
    } = $$props;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      BlockLabel($$renderer3, { show_label, Icon: Video$1, label: label || "Video" });
      $$renderer3.push(`<!----> `);
      if (!value || value.url === void 0) {
        $$renderer3.push("<!--[-->");
        Empty($$renderer3, {
          unpadded_box: true,
          size: "large",
          children: ($$renderer4) => {
            Video$1($$renderer4);
          },
          $$slots: { default: true }
        });
      } else {
        $$renderer3.push("<!--[!-->");
        $$renderer3.push(`<!---->`);
        {
          Player($$renderer3, {
            src: value.url,
            subtitle: subtitle?.url,
            is_stream: value.is_stream,
            autoplay,
            onplay: () => onplay?.(),
            onpause: () => onpause?.(),
            onstop: () => onstop?.(),
            onend: () => onend?.(),
            onloadedmetadata: () => {
              onload?.();
            },
            mirror: false,
            label,
            loop,
            interactive: false,
            upload,
            i18n,
            get playback_position() {
              return playback_position;
            },
            set playback_position($$value) {
              playback_position = $$value;
              $$settled = false;
            }
          });
        }
        $$renderer3.push(`<!----> <div data-testid="download-div">`);
        IconButtonWrapper($$renderer3, {
          display_top_corner: display_icon_button_wrapper_top_corner,
          buttons: buttons ?? ["download", "share"],
          on_custom_button_click,
          children: ($$renderer4) => {
            if (buttons?.some((btn) => typeof btn === "string" && btn === "download")) {
              $$renderer4.push("<!--[-->");
              DownloadLink($$renderer4, {
                href: value.is_stream ? value.url?.replace("playlist.m3u8", "playlist-file") : value.url,
                download: value.orig_name || value.path,
                children: ($$renderer5) => {
                  IconButton($$renderer5, { Icon: Download, label: "Download" });
                },
                $$slots: { default: true }
              });
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]--> `);
            if (buttons?.some((btn) => typeof btn === "string" && btn === "share")) {
              $$renderer4.push("<!--[-->");
              ShareButton($$renderer4, {
                i18n,
                onerror: (detail) => onerror?.(detail),
                onshare: (detail) => onshare?.(detail),
                value,
                formatter: async (value2) => {
                  if (!value2) return "";
                  let url = await uploadToHuggingFace(value2.data);
                  return url;
                }
              });
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]-->`);
          }
        });
        $$renderer3.push(`<!----></div>`);
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { value, playback_position });
  });
}
const VideoPreview$1 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: VideoPreview
}, Symbol.toStringTag, { value: "Module" }));

export { Player as P, VideoPreview as V, playable as a, VideoPreview$1 as b, loaded as l, prettyBytes as p };
//# sourceMappingURL=VideoPreview-BUKiZR1J.js.map
