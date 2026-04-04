import './async-D55cHugf.js';
import { c as spread_props, f as attr_class } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { M as Markdown } from './Index.svelte_svelte_type_style_lang2-y82X8y0w.js';
import { S as Static } from './index3-BSof0MuO.js';
import { B as Block } from './Block-JRVGROlG.js';
import { S as ScrollFade } from './ScrollFade-BvKMcxYM.js';
export { default as BaseExample } from './Example7-CLPLiEhz.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './Check-B-uwlXei.js';
import './Copy-lixG99xU.js';
import './IconButton-D82v5nCM.js';
import './IconButtonWrapper-DvxA4nj6.js';
import './MarkdownCode-BPzQM9w-.js';
import './prism-python-uEphcfi3.js';
import './index50-nVz3eHRG.js';
import 'path';
import 'url';
import 'fs';
import './html-CfyvkLET.js';
import './Clear-DH-TDCgr.js';

function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    let show_fade = false;
    Block($$renderer2, {
      visible: gradio.shared.visible,
      elem_id: gradio.shared.elem_id,
      elem_classes: gradio.shared.elem_classes,
      container: gradio.shared.container,
      allow_overflow: true,
      overflow_behavior: "auto",
      height: gradio.props.height,
      min_height: gradio.props.min_height,
      max_height: gradio.props.max_height,
      rtl: gradio.props.rtl,
      children: ($$renderer3) => {
        Static($$renderer3, spread_props([
          { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
          gradio.shared.loading_status,
          {
            variant: "center",
            on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
          }
        ]));
        $$renderer3.push(`<!----> <div${attr_class("svelte-16ln60g", void 0, {
          "padding": gradio.props.padding,
          "pending": gradio.shared.loading_status?.status === "pending" && gradio.shared.loading_status?.show_progress !== "hidden"
        })}>`);
        Markdown($$renderer3, {
          value: gradio.props.value,
          elem_classes: gradio.shared.elem_classes,
          visible: gradio.shared.visible,
          rtl: gradio.props.rtl,
          onchange: () => gradio.dispatch("change"),
          oncopy: (e) => gradio.dispatch("copy", e.detail),
          latex_delimiters: gradio.props.latex_delimiters,
          sanitize_html: gradio.props.sanitize_html,
          line_breaks: gradio.props.line_breaks,
          header_links: gradio.props.header_links,
          show_copy_button: gradio.props.buttons?.includes("copy"),
          loading_status: gradio.shared.loading_status,
          theme_mode: gradio.shared.theme_mode
        });
        $$renderer3.push(`<!----></div> `);
        ScrollFade($$renderer3, { visible: show_fade });
        $$renderer3.push(`<!---->`);
      },
      $$slots: { default: true }
    });
  });
}

export { Markdown as BaseMarkdown, Index as default };
//# sourceMappingURL=Index12-M-dcVqUC.js.map
