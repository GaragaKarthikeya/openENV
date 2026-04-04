import './async-D55cHugf.js';
import { f as attr_class } from './index-u8mz_F03.js';
import { V as Video } from './Video2-De5bYQY6.js';
import './2-BAeILJ1g.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';

function Example($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { type, selected = false, value = null, loop } = $$props;
    let video = void 0;
    async function init() {
      if (!video) return;
      video.muted = true;
      video.playsInline = true;
      video.controls = false;
      video.setAttribute("muted", "");
      await video.play();
      video.pause();
    }
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (value) {
        $$renderer3.push("<!--[-->");
        {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<div${attr_class("container svelte-1nl1glk", void 0, {
            "table": type === "table",
            "gallery": type === "gallery",
            "selected": selected
          })}>`);
          Video($$renderer3, {
            muted: true,
            playsinline: true,
            onloadeddata: init,
            onmouseover: () => video?.play(),
            onmouseout: () => video?.pause(),
            src: value?.url,
            is_stream: false,
            loop,
            get node() {
              return video;
            },
            set node($$value) {
              video = $$value;
              $$settled = false;
            }
          });
          $$renderer3.push(`<!----></div>`);
        }
        $$renderer3.push(`<!--]-->`);
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
  });
}

export { Example as default };
//# sourceMappingURL=Example28-D1-aV4K8.js.map
