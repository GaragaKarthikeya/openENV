import './async-D55cHugf.js';
import { d as bind_props } from './index-u8mz_F03.js';
import { h as html } from './html-CfyvkLET.js';

/* empty css                                         */
const INLINE_CODE_RE = /`([^`]+)`/g;
const LINK_RE = /\[([^\]]+)\]\(([^)]+)\)/g;
const BOLD_ASTERISK_RE = /\*\*(.+?)\*\*/g;
const BOLD_UNDERSCORE_RE = /__(.+?)__/g;
const ITALIC_ASTERISK_RE = /\*(.+?)\*/g;
const ITALIC_UNDERSCORE_RE = /(?<!\w)_(.+?)_(?!\w)/g;
const PROTOCOL_RE = /^\w+:/;
function escape_html(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function render_link(_match, text, url) {
  const trimmed = url.trim();
  if (PROTOCOL_RE.test(trimmed)) {
    if (/^https?:/i.test(trimmed)) {
      return `<a href="${trimmed}" target="_blank" rel="noopener noreferrer">${text}</a>`;
    }
    return text;
  }
  return `<a href="${trimmed}" target="_blank" rel="noopener noreferrer">${text}</a>`;
}
function render_inline_markdown(text) {
  let result = escape_html(text);
  result = result.replace(INLINE_CODE_RE, "<code>$1</code>");
  result = result.replace(LINK_RE, render_link);
  result = result.replace(BOLD_ASTERISK_RE, "<strong>$1</strong>");
  result = result.replace(BOLD_UNDERSCORE_RE, "<strong>$1</strong>");
  result = result.replace(ITALIC_ASTERISK_RE, "<em>$1</em>");
  result = result.replace(ITALIC_UNDERSCORE_RE, "<em>$1</em>");
  result = result.replace(/\n/g, "<br>");
  return result;
}
function Info($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let info = $$props["info"];
    $$renderer2.push(`<div class="info-text svelte-9hc4ua">${html(render_inline_markdown(info))}</div>`);
    bind_props($$props, { info });
  });
}

export { Info as I };
//# sourceMappingURL=Info-BeS-ygt-.js.map
