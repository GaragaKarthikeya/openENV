import './async-D55cHugf.js';
import { d as bind_props, g as attr_style, f as attr_class, i as stringify, e as ensure_array_like, a as attr } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { t as tick } from './index-server-CQz6EZl_.js';
import './Code.svelte_svelte_type_style_lang-wttD6sHm.js';
import { M as Markdown } from './Index.svelte_svelte_type_style_lang2-y82X8y0w.js';
import './MarkdownCode-BPzQM9w-.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './utils.svelte-aYomt3KL.js';
import './clone-Yk88IHKV.js';
import './Check-B-uwlXei.js';
import './Copy-lixG99xU.js';
import './IconButton-D82v5nCM.js';
import './IconButtonWrapper-DvxA4nj6.js';
import './prism-python-uEphcfi3.js';
import './index50-nVz3eHRG.js';
import 'path';
import 'url';
import 'fs';
import './html-CfyvkLET.js';

/* empty css                                       */
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let app = $$props["app"];
    let root = $$props["root"];
    let prompt = "";
    let editorWidth = 350;
    let activeTab = "chat";
    let codeValue = "";
    let message_history = [];
    const scroll_to_bottom = (behavior = "smooth") => {
      return;
    };
    let starterQueries = [];
    const fetchStarterQueries = async () => {
      const post = app.post_data(`${root}/gradio_api/vibe-starter-queries/`, {});
      post.then(async ([response, status_code]) => {
        if (status_code !== 200) {
          throw new Error(`Error: ${status_code}`);
        }
        const responseData = response;
        starterQueries = responseData.starter_queries;
      }).catch(async (error) => {
        console.error("Failed to fetch starter queries:", error);
      });
    };
    fetchStarterQueries();
    const fetchCode = async () => {
      try {
        const response = await fetch(`${root}/gradio_api/vibe-code/`, {
          method: "GET",
          headers: { "Content-Type": "application/json" }
        });
        if (response.ok) {
          const data = await response.json();
          codeValue = data.code;
        }
      } catch (error) {
        console.error("Failed to fetch code:", error);
      }
    };
    fetchCode();
    {
      tick().then(() => scroll_to_bottom("auto"));
    }
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div class="vibe-editor svelte-1s2fnws"${attr_style(`width: ${stringify(editorWidth)}px;`)}><button class="resize-handle svelte-1s2fnws" aria-label="Resize sidebar"></button> <div class="tab-header svelte-1s2fnws"><button${attr_class("tab-button svelte-1s2fnws", void 0, { "active": activeTab === "chat" })}>Chat</button> <button${attr_class("tab-button svelte-1s2fnws", void 0, { "active": activeTab === "code" })}>Code `);
      {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--></button></div> <div class="tab-content svelte-1s2fnws">`);
      {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<div class="message-history svelte-1s2fnws"><!--[-->`);
        const each_array = ensure_array_like(message_history);
        for (let index = 0, $$length = each_array.length; index < $$length; index++) {
          let message = each_array[index];
          $$renderer3.push(`<div${attr_class("message-item svelte-1s2fnws", void 0, { "bot-message": message.isBot, "user-message": !message.isBot })}><div class="message-content svelte-1s2fnws"><span class="message-text svelte-1s2fnws">`);
          Markdown($$renderer3, {
            value: message.text,
            latex_delimiters: [],
            theme_mode: "system"
          });
          $$renderer3.push(`<!----></span> `);
          if (!message.isBot && message.hash && !message.isPending) {
            $$renderer3.push("<!--[-->");
            $$renderer3.push(`<button class="undo-button svelte-1s2fnws" title="Undo this change">Undo</button>`);
          } else {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]--></div></div>`);
        }
        $$renderer3.push(`<!--]--> `);
        if (message_history.length === 0) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<div class="no-messages svelte-1s2fnws">No messages yet</div>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        if (message_history.length === 0) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<div class="starter-queries-container svelte-1s2fnws"><div class="starter-queries svelte-1s2fnws"><!--[-->`);
          const each_array_1 = ensure_array_like(starterQueries);
          for (let $$index_1 = 0, $$length = each_array_1.length; $$index_1 < $$length; $$index_1++) {
            let query = each_array_1[$$index_1];
            $$renderer3.push(`<button class="starter-query-button svelte-1s2fnws">${escape_html(query)}</button>`);
          }
          $$renderer3.push(`<!--]--></div></div>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--></div>`);
      }
      $$renderer3.push(`<!--]--></div> <div class="input-section svelte-1s2fnws"><div class="powered-by svelte-1s2fnws">Powered by: <code>gpt-oss</code></div> <textarea placeholder="What can I add or change?" class="prompt-input svelte-1s2fnws">`);
      const $$body = escape_html(prompt);
      if ($$body) {
        $$renderer3.push(`${$$body}`);
      }
      $$renderer3.push(`</textarea> <button class="submit-button svelte-1s2fnws"${attr("disabled", prompt.trim() === "", true)}>Send</button></div></div>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { app, root });
  });
}

export { Index as default };
//# sourceMappingURL=Index18-DNsben2n.js.map
