import './async-D55cHugf.js';
import { f as attr_class, d as bind_props } from './index-u8mz_F03.js';
import { U as Upload, I as ImagePaste } from './Upload-CIQ-D6yx.js';
import { M as Microphone } from './Microphone-CCAKTpuQ.js';
import { V as Video } from './Video-_1zl9-Cr.js';
import { W as Webcam } from './Webcam-iV1BaoQQ.js';

function SelectSource($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      sources,
      active_source = void 0,
      handle_clear = () => {
      },
      handle_select = () => {
      }
    } = $$props;
    let unique_sources = [...new Set(sources)];
    if (unique_sources.length > 1 || sources.includes("clipboard")) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<span class="source-selection svelte-exvkcd" data-testid="source-select">`);
      if (sources.includes("upload")) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<button${attr_class("icon svelte-exvkcd", void 0, { "selected": active_source === "upload" || !active_source })} aria-label="Upload file">`);
        Upload($$renderer2);
        $$renderer2.push(`<!----></button>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (sources.includes("microphone")) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<button${attr_class("icon svelte-exvkcd", void 0, { "selected": active_source === "microphone" })} aria-label="Record audio">`);
        Microphone($$renderer2);
        $$renderer2.push(`<!----></button>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (sources.includes("webcam")) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<button${attr_class("icon svelte-exvkcd", void 0, { "selected": active_source === "webcam" })} aria-label="Capture from camera">`);
        Webcam($$renderer2);
        $$renderer2.push(`<!----></button>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (sources.includes("webcam-video")) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<button${attr_class("icon svelte-exvkcd", void 0, { "selected": active_source === "webcam-video" })} aria-label="Record video from camera">`);
        Video($$renderer2);
        $$renderer2.push(`<!----></button>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (sources.includes("clipboard")) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<button${attr_class("icon svelte-exvkcd", void 0, { "selected": active_source === "clipboard" })} aria-label="Paste from clipboard">`);
        ImagePaste($$renderer2);
        $$renderer2.push(`<!----></button>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></span>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { active_source });
  });
}

export { SelectSource as S };
//# sourceMappingURL=SelectSource-CeUXfvCc.js.map
