import { f as fallback } from './async-D55cHugf.js';
import { n as sanitize_props, r as rest_props, m as attributes, s as slot, d as bind_props } from './index-u8mz_F03.js';

/* empty css                                         */
function DownloadLink($$renderer, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, ["href", "download"]);
  $$renderer.component(($$renderer2) => {
    let href = fallback($$props["href"], void 0);
    let download = $$props["download"];
    $$renderer2.push(`<a${attributes(
      {
        class: "download-link",
        href,
        target: typeof window !== "undefined" && window.__is_colab__ ? "_blank" : null,
        rel: "noopener noreferrer",
        download,
        ...$$restProps
      },
      "svelte-7nkusa",
      void 0,
      { position: "relative" }
    )}><!--[-->`);
    slot($$renderer2, $$props, "default", {});
    $$renderer2.push(`<!--]--></a>`);
    bind_props($$props, { href, download });
  });
}

export { DownloadLink as D };
//# sourceMappingURL=DownloadLink-BMNxSkQ8.js.map
