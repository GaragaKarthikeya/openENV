import { f as fallback } from './async-D55cHugf.js';
import { f as attr_class, d as bind_props } from './index-u8mz_F03.js';
import { I as ImagePaste, U as Upload } from './Upload-CIQ-D6yx.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';

const RE_HEADING = /^(#\s*)(.+)$/m;
function inject(text) {
  const trimmed_text = text.trim();
  const heading_match = trimmed_text.match(RE_HEADING);
  if (!heading_match) {
    return [false, trimmed_text || false];
  }
  const [full_match, , heading_content] = heading_match;
  const _heading = heading_content.trim();
  if (trimmed_text === full_match) {
    return [_heading, false];
  }
  const heading_end_index = heading_match.index !== void 0 ? heading_match.index + full_match.length : 0;
  const remaining_text = trimmed_text.substring(heading_end_index).trim();
  const _paragraph = remaining_text || false;
  return [_heading, _paragraph];
}
function UploadText($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let heading, paragraph;
    let type = fallback($$props["type"], "file");
    let i18n = $$props["i18n"];
    let message = fallback($$props["message"], void 0);
    let mode = fallback($$props["mode"], "full");
    let hovered = fallback($$props["hovered"], false);
    let placeholder = fallback($$props["placeholder"], void 0);
    const defs = {
      image: "upload_text.drop_image",
      video: "upload_text.drop_video",
      audio: "upload_text.drop_audio",
      file: "upload_text.drop_file",
      csv: "upload_text.drop_csv",
      gallery: "upload_text.drop_gallery",
      clipboard: "upload_text.paste_clipboard"
    };
    [heading, paragraph] = placeholder ? inject(placeholder) : [false, false];
    $$renderer2.push(`<div class="wrap svelte-1vmd51o"><span${attr_class("icon-wrap svelte-1vmd51o", void 0, { "hovered": hovered })}>`);
    if (type === "clipboard") {
      $$renderer2.push("<!--[-->");
      ImagePaste($$renderer2);
    } else {
      $$renderer2.push("<!--[!-->");
      Upload($$renderer2);
    }
    $$renderer2.push(`<!--]--></span> `);
    if (heading || paragraph) {
      $$renderer2.push("<!--[-->");
      if (heading) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<h2 class="svelte-1vmd51o">${escape_html(heading)}</h2>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (paragraph) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<p class="svelte-1vmd51o">${escape_html(paragraph)}</p>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    } else {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`${escape_html(i18n(defs[type] || defs.file))} `);
      if (mode !== "short") {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<span class="or svelte-1vmd51o">- ${escape_html(i18n("common.or"))} -</span> ${escape_html(message || i18n("upload_text.click_to_upload"))}`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></div>`);
    bind_props($$props, { type, i18n, message, mode, hovered, placeholder });
  });
}

export { UploadText as U };
//# sourceMappingURL=UploadText-CnLI0lM1.js.map
