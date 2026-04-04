import './async-D55cHugf.js';
import { f as attr_class, i as stringify, d as bind_props, c as spread_props } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { S as Static } from './index3-BSof0MuO.js';
import { C as Code$1 } from './Code-DcA0iOIn.js';
import { o as onDestroy } from './index-server-CQz6EZl_.js';
import { E as EditorView, R as RangeValue } from './Code.svelte_svelte_type_style_lang-wttD6sHm.js';
import { C as Check } from './Check-B-uwlXei.js';
import { C as Copy } from './Copy-lixG99xU.js';
import { I as IconButton } from './IconButton-D82v5nCM.js';
import { D as Download } from './Download-ByiErn53.js';
import { D as DownloadLink } from './DownloadLink-BMNxSkQ8.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { B as Block } from './Block-JRVGROlG.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { E as Empty } from './Empty-BHvMDNv5.js';
export { default as BaseExample } from './Example-DQafirJP.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './Clear-DH-TDCgr.js';

const closedBracket = /* @__PURE__ */ new class extends RangeValue {
}();
closedBracket.startSide = 1;
closedBracket.endSide = -1;
function Code($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      class_names = "",
      value = void 0,
      dark_mode,
      basic = true,
      language,
      lines = 5,
      max_lines = null,
      extensions = [],
      use_tab = true,
      readonly = false,
      placeholder: placeholder$1 = void 0,
      wrap_lines = false,
      show_line_numbers = true,
      autocomplete = false,
      onchange,
      onblur,
      onfocus,
      oninput
    } = $$props;
    EditorView.theme({
      "&": {
        fontSize: "var(--text-sm)",
        backgroundColor: "var(--border-color-secondary)"
      },
      ".cm-content": {
        paddingTop: "5px",
        paddingBottom: "5px",
        color: "var(--body-text-color)",
        fontFamily: "var(--font-mono)",
        minHeight: "100%"
      },
      ".cm-gutterElement": { marginRight: "var(--spacing-xs)" },
      ".cm-gutters": {
        marginRight: "1px",
        borderRight: "1px solid var(--border-color-primary)",
        backgroundColor: "var(--block-background-fill);",
        color: "var(--body-text-color-subdued)"
      },
      ".cm-focused": { outline: "none" },
      ".cm-scroller": { height: "auto" },
      ".cm-cursor": { borderLeftColor: "var(--body-text-color)" }
    });
    EditorView.theme({
      ".cm-tooltip-autocomplete": {
        "& > ul": {
          backgroundColor: "var(--background-fill-primary)",
          color: "var(--body-text-color)"
        },
        "& > ul > li[aria-selected]": {
          backgroundColor: "var(--color-accent-soft)",
          color: "var(--body-text-color)"
        }
      }
    });
    $$renderer2.push(`<div class="wrap svelte-pndh99"><div${attr_class(`codemirror-wrapper ${stringify(class_names)}`, "svelte-pndh99")}></div></div>`);
    bind_props($$props, { value });
  });
}
function Copy_1($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { value } = $$props;
    let copied = false;
    let timer;
    function copy_feedback() {
      copied = true;
      if (timer) clearTimeout(timer);
      timer = setTimeout(
        () => {
          copied = false;
        },
        2e3
      );
    }
    async function handle_copy() {
      if ("clipboard" in navigator) {
        await navigator.clipboard.writeText(value);
        copy_feedback();
      }
    }
    onDestroy(() => {
      if (timer) clearTimeout(timer);
    });
    IconButton($$renderer2, { Icon: copied ? Check : Copy, onclick: handle_copy });
  });
}
function Download_1($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { value, language } = $$props;
    let ext = get_ext_for_type(language);
    function get_ext_for_type(type) {
      const exts = {
        py: "py",
        python: "py",
        md: "md",
        markdown: "md",
        json: "json",
        html: "html",
        css: "css",
        js: "js",
        javascript: "js",
        ts: "ts",
        typescript: "ts",
        yaml: "yaml",
        yml: "yml",
        dockerfile: "dockerfile",
        sh: "sh",
        shell: "sh",
        r: "r",
        c: "c",
        cpp: "cpp",
        latex: "tex"
      };
      return exts[type] || "txt";
    }
    let copied = false;
    let timer;
    function copy_feedback() {
      copied = true;
      if (timer) clearTimeout(timer);
      timer = setTimeout(
        () => {
          copied = false;
        },
        2e3
      );
    }
    let download_value = URL.createObjectURL(new Blob([value]));
    onDestroy(() => {
      if (timer) clearTimeout(timer);
    });
    DownloadLink($$renderer2, {
      download: `file.${stringify(ext)}`,
      href: download_value,
      onclick: copy_feedback,
      children: ($$renderer3) => {
        IconButton($$renderer3, { Icon: copied ? Check : Download });
      },
      $$slots: { default: true }
    });
  });
}
function Widgets($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value,
      language,
      buttons = null,
      on_custom_button_click = null
    } = $$props;
    IconButtonWrapper($$renderer2, {
      buttons,
      on_custom_button_click,
      children: ($$renderer3) => {
        if (buttons?.some((btn) => typeof btn === "string" && btn === "download")) {
          $$renderer3.push("<!--[-->");
          Download_1($$renderer3, { value, language });
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        if (buttons?.some((btn) => typeof btn === "string" && btn === "copy")) {
          $$renderer3.push("<!--[-->");
          Copy_1($$renderer3, { value });
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]-->`);
      }
    });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    let dark_mode = gradio.shared.theme === "dark";
    let label = gradio.shared.label || gradio.i18n("code.code");
    gradio.props.value;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      Block($$renderer3, {
        height: gradio.props.max_lines && "fit-content",
        variant: "solid",
        padding: false,
        elem_id: gradio.shared.elem_id,
        elem_classes: gradio.shared.elem_classes,
        visible: gradio.shared.visible,
        scale: gradio.shared.scale,
        min_width: gradio.shared.min_width,
        children: ($$renderer4) => {
          Static($$renderer4, spread_props([
            { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
            gradio.shared.loading_status,
            {
              on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
            }
          ]));
          $$renderer4.push(`<!----> `);
          if (gradio.shared.show_label) {
            $$renderer4.push("<!--[-->");
            BlockLabel($$renderer4, {
              Icon: Code$1,
              show_label: gradio.shared.show_label,
              label,
              float: false
            });
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]--> `);
          if (!gradio.props.value && !gradio.shared.interactive) {
            $$renderer4.push("<!--[-->");
            Empty($$renderer4, {
              unpadded_box: true,
              size: "large",
              children: ($$renderer5) => {
                Code$1($$renderer5);
              },
              $$slots: { default: true }
            });
          } else {
            $$renderer4.push("<!--[!-->");
            Widgets($$renderer4, {
              language: gradio.props.language,
              value: gradio.props.value,
              buttons: gradio.props.buttons ?? ["copy", "download"],
              on_custom_button_click: (id) => {
                gradio.dispatch("custom_button_click", { id });
              }
            });
            $$renderer4.push(`<!----> `);
            Code($$renderer4, {
              language: gradio.props.language,
              lines: gradio.props.lines,
              max_lines: gradio.props.max_lines,
              dark_mode,
              wrap_lines: gradio.props.wrap_lines,
              show_line_numbers: gradio.props.show_line_numbers,
              autocomplete: gradio.props.autocomplete,
              readonly: !gradio.shared.interactive,
              onblur: () => gradio.dispatch("blur"),
              onfocus: () => gradio.dispatch("focus"),
              oninput: () => gradio.dispatch("input"),
              get value() {
                return gradio.props.value;
              },
              set value($$value) {
                gradio.props.value = $$value;
                $$settled = false;
              }
            });
            $$renderer4.push(`<!---->`);
          }
          $$renderer4.push(`<!--]-->`);
        },
        $$slots: { default: true }
      });
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
  });
}

export { Code as BaseCode, Copy_1 as BaseCopy, Download_1 as BaseDownload, Widgets as BaseWidget, Index as default };
//# sourceMappingURL=Index52-Co7gLzPI.js.map
