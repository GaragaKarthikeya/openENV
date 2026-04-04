import './async-D55cHugf.js';
import { g as attr_style, a as attr, e as ensure_array_like, i as stringify, f as attr_class } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { P as Prism$1 } from './prism-python-uEphcfi3.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import { h as html } from './html-CfyvkLET.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';

var prismTypescript = {};
var hasRequiredPrismTypescript;
function requirePrismTypescript() {
  if (hasRequiredPrismTypescript) return prismTypescript;
  hasRequiredPrismTypescript = 1;
  (function(Prism2) {
    Prism2.languages.typescript = Prism2.languages.extend("javascript", {
      "class-name": {
        pattern: /(\b(?:class|extends|implements|instanceof|interface|new|type)\s+)(?!keyof\b)(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*(?:\s*<(?:[^<>]|<(?:[^<>]|<[^<>]*>)*>)*>)?/,
        lookbehind: true,
        greedy: true,
        inside: null
        // see below
      },
      "builtin": /\b(?:Array|Function|Promise|any|boolean|console|never|number|string|symbol|unknown)\b/
    });
    Prism2.languages.typescript.keyword.push(
      /\b(?:abstract|declare|is|keyof|readonly|require)\b/,
      // keywords that have to be followed by an identifier
      /\b(?:asserts|infer|interface|module|namespace|type)\b(?=\s*(?:[{_$a-zA-Z\xA0-\uFFFF]|$))/,
      // This is for `import type *, {}`
      /\btype\b(?=\s*(?:[\{*]|$))/
    );
    delete Prism2.languages.typescript["parameter"];
    delete Prism2.languages.typescript["literal-property"];
    var typeInside = Prism2.languages.extend("typescript", {});
    delete typeInside["class-name"];
    Prism2.languages.typescript["class-name"].inside = typeInside;
    Prism2.languages.insertBefore("typescript", "function", {
      "decorator": {
        pattern: /@[$\w\xA0-\uFFFF]+/,
        inside: {
          "at": {
            pattern: /^@/,
            alias: "operator"
          },
          "function": /^[\s\S]+/
        }
      },
      "generic-function": {
        // e.g. foo<T extends "bar" | "baz">( ...
        pattern: /#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*\s*<(?:[^<>]|<(?:[^<>]|<[^<>]*>)*>)*>(?=\s*\()/,
        greedy: true,
        inside: {
          "function": /^#?(?!\s)[_$a-zA-Z\xA0-\uFFFF](?:(?!\s)[$\w\xA0-\uFFFF])*/,
          "generic": {
            pattern: /<[\s\S]+/,
            // everything after the first <
            alias: "class-name",
            inside: typeInside
          }
        }
      }
    });
    Prism2.languages.ts = Prism2.languages.typescript;
  })(Prism);
  return prismTypescript;
}
requirePrismTypescript();
function ParamViewer($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { docs, linkify = [], header, anchor_links, max_height } = $$props;
    let lang = "python";
    let _docs = highlight_code(docs, lang);
    function create_slug(name, anchor_links2) {
      let prefix = "param-";
      if (typeof anchor_links2 === "string") {
        prefix += anchor_links2 + "-";
      }
      return prefix + name.toLowerCase().replace(/[^a-z0-9]+/g, "-");
    }
    function highlight(code, lang2) {
      let highlighted = Prism$1.highlight(code, Prism$1.languages[lang2], lang2);
      for (const link of linkify) {
        highlighted = highlighted.replace(new RegExp(link, "g"), `<a href="#h-${link.toLocaleLowerCase()}">${link}</a>`);
      }
      return highlighted;
    }
    function highlight_code(_docs2, lang2) {
      if (!_docs2) {
        return [];
      }
      return Object.entries(_docs2).map(([name, { type, description, default: _default }]) => {
        let highlighted_type = type ? highlight(type, lang2) : null;
        return {
          name,
          type: highlighted_type,
          description,
          default: _default ? highlight(_default, lang2) : null
        };
      });
    }
    function render_links(description) {
      const escaped = description.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
      const markdown_links = escaped.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
      return markdown_links;
    }
    const get_dimension = (dimension_value) => {
      if (dimension_value === void 0) {
        return void 0;
      }
      if (typeof dimension_value === "number") {
        return dimension_value + "px";
      } else if (typeof dimension_value === "string") {
        return dimension_value;
      }
    };
    $$renderer2.push(`<div class="wrap svelte-1kuiw39"${attr_style("", { "max-height": get_dimension(max_height) })}><div class="header svelte-1kuiw39">`);
    if (header !== null) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<span class="title svelte-1kuiw39">${escape_html(header)}</span>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> <button class="toggle-all svelte-1kuiw39"${attr("title", "Open All")}>${escape_html("▼")}</button></div> `);
    if (_docs) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="param-content svelte-1kuiw39"><!--[-->`);
      const each_array = ensure_array_like(_docs);
      for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
        let { type, description, default: _default, name } = each_array[$$index];
        $$renderer2.push(`<details class="param md svelte-1kuiw39"${attr("id", anchor_links ? create_slug(name || "", anchor_links) : void 0)}><summary class="type svelte-1kuiw39">`);
        if (anchor_links) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<a${attr("href", `#${stringify(create_slug(name || "", anchor_links))}`)} class="param-link svelte-1kuiw39"><span class="link-icon svelte-1kuiw39">🔗</span></a>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]--> <pre${attr_class(`language-${stringify(lang)}`, "svelte-1kuiw39")}><code class="svelte-1kuiw39">${escape_html(name)}`);
        if (type) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`: ${html(type)}`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]--></code></pre></summary> `);
        if (_default) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<div${attr_class("default svelte-1kuiw39", void 0, { "last": !description })}><span class="svelte-1kuiw39"${attr_style("", { "padding-right": "4px" })}>default</span> <code class="svelte-1kuiw39">= ${html(_default)}</code></div>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]--> `);
        if (description) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<div class="description svelte-1kuiw39"><p>${html(render_links(description))}</p></div>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]--></details>`);
      }
      $$renderer2.push(`<!--]--></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    ParamViewer($$renderer2, {
      docs: gradio.props.value,
      linkify: gradio.props.linkify,
      header: gradio.props.header,
      anchor_links: gradio.props.anchor_links,
      max_height: gradio.props.max_height
    });
  });
}

export { Index as default };
//# sourceMappingURL=Index49-BZTDsGNu.js.map
