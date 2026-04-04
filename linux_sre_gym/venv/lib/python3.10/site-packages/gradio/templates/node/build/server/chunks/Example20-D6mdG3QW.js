import { f as fallback } from './async-D55cHugf.js';
import { f as attr_class, a as attr, d as bind_props } from './index-u8mz_F03.js';
import './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';

function Example($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let value = $$props["value"];
    let samples_dir = $$props["samples_dir"];
    let type = $$props["type"];
    let selected = fallback($$props["selected"], false);
    $$renderer2.push(`<div${attr_class("wrap svelte-ulqlw7", void 0, {
      "table": type === "table",
      "gallery": type === "gallery",
      "selected": selected
    })}><img${attr("src", samples_dir + value[0])} class="svelte-ulqlw7"/> <img${attr("src", samples_dir + value[1])} class="svelte-ulqlw7"/> <span class="svelte-ulqlw7"></span></div>`);
    bind_props($$props, { value, samples_dir, type, selected });
  });
}

export { Example as default };
//# sourceMappingURL=Example20-D6mdG3QW.js.map
