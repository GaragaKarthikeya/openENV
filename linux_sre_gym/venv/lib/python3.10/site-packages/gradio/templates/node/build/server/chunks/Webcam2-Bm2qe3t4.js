import { f as fallback } from './async-D55cHugf.js';
import { f as attr_class, a as attr, d as bind_props, g as attr_style } from './index-u8mz_F03.js';
import { o as onDestroy } from './index-server-CQz6EZl_.js';
import { a as prepare_files } from './2-BAeILJ1g.js';
import { S as StreamingBar } from './StreamingBar-B1BHh5xK.js';
import { W as Webcam$1 } from './Webcam-iV1BaoQQ.js';

/* empty css                                            */
function WebcamPermissions($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    $$renderer2.push(`<button class="svelte-1dqolfz"${attr_style("", { height: "100%" })}><div class="wrap svelte-1dqolfz"><span class="icon-wrap svelte-1dqolfz">`);
    Webcam$1($$renderer2);
    $$renderer2.push(`<!----></span> Click to Access Webcam</div></button>`);
  });
}
function Webcam($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let video_source;
    let stream_state = fallback($$props["stream_state"], "closed");
    let canvas;
    let streaming = fallback($$props["streaming"], false);
    let pending = fallback($$props["pending"], false);
    let root = fallback($$props["root"], "");
    let stream_every = fallback($$props["stream_every"], 1);
    let mode = fallback($$props["mode"], "image");
    let mirror_webcam = $$props["mirror_webcam"];
    let include_audio = $$props["include_audio"];
    let webcam_constraints = fallback($$props["webcam_constraints"], null);
    let i18n = $$props["i18n"];
    let upload = $$props["upload"];
    let value = fallback($$props["value"], null);
    let time_limit = fallback($$props["time_limit"], null);
    function take_picture() {
      if ((!streaming || streaming && recording) && video_source.videoWidth && video_source.videoHeight) {
        var context = canvas.getContext("2d");
        canvas.width = video_source.videoWidth;
        canvas.height = video_source.videoHeight;
        context.drawImage(video_source, 0, 0, video_source.videoWidth, video_source.videoHeight);
        if (mirror_webcam) {
          context.scale(-1, 1);
          context.drawImage(video_source, -video_source.videoWidth, 0);
        }
        if (streaming && (!recording || stream_state === "waiting")) {
          return;
        }
        if (streaming) {
          canvas.toDataURL("image/jpeg");
          return;
        }
        canvas.toBlob(
          (blob) => {
          },
          `image/${streaming ? "jpeg" : "png"}`,
          0.8
        );
      }
    }
    let recording = false;
    let recorded_blobs = [];
    let stream;
    let mimeType;
    let media_recorder;
    function take_recording() {
      if (recording) {
        media_recorder.stop();
        let video_blob = new Blob(recorded_blobs, { type: mimeType });
        let ReaderObj = new FileReader();
        ReaderObj.onload = async function(e) {
          if (e.target) {
            let _video_blob = new File([video_blob], "sample." + mimeType.substring(6));
            const val = await prepare_files([_video_blob]);
            (await upload(val, root))?.filter(Boolean)[0];
          }
        };
        ReaderObj.readAsDataURL(video_blob);
      } else if (typeof MediaRecorder !== "undefined") {
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
        media_recorder.addEventListener("dataavailable", function(e) {
          recorded_blobs.push(e.data);
        });
        media_recorder.start(200);
      }
      recording = !recording;
    }
    function record_video_or_photo({ destroy } = {}) {
      if (mode === "image" && streaming) {
        recording = !recording;
      }
      if (!destroy) {
        if (mode === "image") {
          take_picture();
        } else {
          take_recording();
        }
      }
    }
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
    onDestroy(() => {
      if (typeof window === "undefined") return;
      record_video_or_photo({ destroy: true });
    });
    $$renderer2.push(`<div class="wrap svelte-1tktvmr">`);
    StreamingBar($$renderer2, { time_limit });
    $$renderer2.push(`<!---->  <video${attr_class("svelte-1tktvmr", void 0, {
      "flip": mirror_webcam,
      "hide": true
    })}></video> <img${attr("src", value?.url)}${attr_class("svelte-1tktvmr", void 0, { "hide": true })}/> `);
    {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div title="grant webcam access" style="height: 100%">`);
      WebcamPermissions($$renderer2);
      $$renderer2.push(`<!----></div>`);
    }
    $$renderer2.push(`<!--]--></div>`);
    bind_props($$props, {
      stream_state,
      streaming,
      pending,
      root,
      stream_every,
      mode,
      mirror_webcam,
      include_audio,
      webcam_constraints,
      i18n,
      upload,
      value,
      time_limit,
      click_outside
    });
  });
}

export { Webcam as W };
//# sourceMappingURL=Webcam2-Bm2qe3t4.js.map
