import { p as prop, e as init } from './i18n-CMo5lpzy.js';
import { R as push, G as deep_read_state, z as untrack, a as append, T as pop, U as flushSync, V as child, W as from_html, Y as reset } from './index-BvBk1Iap.js';
import { h as html } from './html-CdUEtR5E.js';
import './ScrollFade.svelte_svelte_type_style_lang-Xm-PLg9E.js';

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

var root = from_html(`<div class="info-text svelte-9hc4ua"><!></div>`);

function Info($$anchor, $$props) {
	push($$props, false);

	let info = prop($$props, 'info', 12);

	var $$exports = {
		get info() {
			return info();
		},

		set info($$value) {
			info($$value);
			flushSync();
		}
	};

	init();

	var div = root();
	var node = child(div);

	html(node, () => (
		deep_read_state(render_inline_markdown),
		deep_read_state(info()),
		untrack(() => render_inline_markdown(info()))
	));

	reset(div);
	append($$anchor, div);

	return pop($$exports);
}

export { Info as I };
//# sourceMappingURL=Info-BrkJhX6_.js.map
