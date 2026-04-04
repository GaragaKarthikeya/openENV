import { f as fallback } from './async-D55cHugf.js';
import { f as attr_class, e as ensure_array_like, g as attr_style, i as stringify, a as attr, d as bind_props, c as spread_props } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { B as BlockTitle } from './BlockTitle-B0onSZmA.js';
import { I as IconButton } from './IconButton-D82v5nCM.js';
import { C as Check } from './Check-B-uwlXei.js';
import { C as Copy } from './Copy-lixG99xU.js';
import { S as Send } from './Send-zUFiC8KE.js';
import { T as Trash } from './Trash-DSMnxSso.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { b as Dropdown } from './Dropdown-BhyRR6a4.js';
import { B as Block } from './Block-JRVGROlG.js';
import { S as Static } from './index3-BSof0MuO.js';
export { default as BaseExample } from './Example15-Xubv-cAE.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './Info-BeS-ygt-.js';
import './html-CfyvkLET.js';
import './Clear-DH-TDCgr.js';

function Plus($$renderer) {
  $$renderer.push(`<svg width="100%" height="100%" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M6 12H12M18 12H12M12 12V6M12 12V18" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></svg>`);
}

function Switch($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let label = $$props["label"];
    let checked = fallback($$props["checked"], false);
    let disabled = fallback($$props["disabled"], false);
    $$renderer2.push(`<div class="s s--slider svelte-1io85vl" style="font-size:var(--font-size-sm)px"><button role="switch"${attr("aria-checked", checked)}${attr("disabled", disabled, true)} class="svelte-1io85vl"></button> <span>${escape_html(label)}</span></div>`);
    bind_props($$props, { label, checked, disabled });
  });
}
function Dialogue($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = props.gradio;
    let checked = false;
    let disabled = !gradio.shared.interactive;
    let dialogue_lines = [];
    let buttons = gradio.props.buttons || ["copy"];
    let on_custom_button_click = (id) => {
      gradio.dispatch("custom_button_click", { id });
    };
    gradio.props.value;
    let copied = false;
    let timer;
    let textbox_value = "";
    let hoveredSpeaker = null;
    let is_unformatting = false;
    let is_formatting = false;
    const defaultColorNames = [
      "red",
      "green",
      "blue",
      "yellow",
      "purple",
      "teal",
      "orange",
      "cyan",
      "lime",
      "pink"
    ];
    const colorNameToHex = {
      red: "rgba(254, 202, 202, 0.7)",
      green: "rgba(209, 250, 229, 0.7)",
      blue: "rgba(219, 234, 254, 0.7)",
      yellow: "rgba(254, 243, 199, 0.7)",
      purple: "rgba(233, 213, 255, 0.7)",
      teal: "rgba(204, 251, 241, 0.7)",
      orange: "rgba(254, 215, 170, 0.7)",
      cyan: "rgba(207, 250, 254, 0.7)",
      lime: "rgba(217, 249, 157, 0.7)",
      pink: "rgba(252, 231, 243, 0.7)"
    };
    let speakerColors = (() => {
      let _speakerColors = {};
      if (gradio.props.color_map) {
        _speakerColors = { ...gradio.props.color_map };
      } else {
        _speakerColors = {};
        gradio.props.speakers.forEach((speaker, index) => {
          const colorName = defaultColorNames[index % defaultColorNames.length];
          _speakerColors[speaker] = colorNameToHex[colorName];
        });
      }
      return _speakerColors;
    })();
    function update_line(index, key, value) {
      dialogue_lines[index][key] = value;
      dialogue_lines = [...dialogue_lines];
      gradio.props.value = [...dialogue_lines];
    }
    async function value_to_string(value) {
      if (typeof value === "string") {
        return value;
      }
      return await gradio.shared.server.format(value);
    }
    async function handle_copy() {
      if ("clipboard" in navigator) {
        const text = await value_to_string(gradio.props.value);
        await navigator.clipboard.writeText(text);
        copy_feedback();
      }
    }
    function copy_feedback() {
      copied = true;
      if (timer) clearTimeout(timer);
      timer = setTimeout(
        () => {
          copied = false;
        },
        1e3
      );
    }
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      $$renderer3.push(`<div${attr_class("svelte-1p54cvv", void 0, { "container": gradio.shared.container })}>`);
      if (gradio.shared.show_label && (buttons.some((btn) => typeof btn === "string" && btn === "copy") || buttons.some((btn) => typeof btn !== "string"))) {
        $$renderer3.push("<!--[-->");
        IconButtonWrapper($$renderer3, {
          buttons,
          on_custom_button_click,
          children: ($$renderer4) => {
            if (buttons.some((btn) => typeof btn === "string" && btn === "copy")) {
              $$renderer4.push("<!--[-->");
              IconButton($$renderer4, {
                Icon: copied ? Check : Copy,
                onclick: handle_copy,
                label: copied ? "Copied" : "Copy"
              });
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]-->`);
          }
        });
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      BlockTitle($$renderer3, {
        show_label: gradio.shared.show_label,
        info: gradio.props.info,
        children: ($$renderer4) => {
          $$renderer4.push(`<!---->${escape_html(gradio.shared.label || "Dialogue")}`);
        },
        $$slots: { default: true }
      });
      $$renderer3.push(`<!----> `);
      if (gradio.props.ui_mode === "both") {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<div${attr_class("switch-container top-switch svelte-1p54cvv", void 0, { "switch-disabled": is_unformatting })}>`);
        Switch($$renderer3, {
          label: "Plain Text",
          disabled: is_unformatting,
          get checked() {
            return checked;
          },
          set checked($$value) {
            checked = $$value;
            $$settled = false;
          }
        });
        $$renderer3.push(`<!----></div>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      if (!checked && gradio.props.ui_mode !== "text") {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<div${attr_class("dialogue-container svelte-1p54cvv", void 0, { "loading": is_unformatting })}>`);
        {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> <!--[-->`);
        const each_array = ensure_array_like(dialogue_lines);
        for (let i = 0, $$length = each_array.length; i < $$length; i++) {
          let line = each_array[i];
          $$renderer3.push(`<div class="dialogue-line svelte-1p54cvv"${attr_style(`--speaker-bg-color: ${stringify(disabled && hoveredSpeaker === null ? speakerColors[line.speaker] || "transparent" : "transparent")}`)}><div class="speaker-column svelte-1p54cvv" role="button" tabindex="0">`);
          if (disabled) {
            $$renderer3.push("<!--[-->");
            $$renderer3.push(`<textarea${attr("disabled", disabled, true)} rows="1" readonly class="svelte-1p54cvv">`);
            const $$body = escape_html(line.speaker);
            if ($$body) {
              $$renderer3.push(`${$$body}`);
            }
            $$renderer3.push(`</textarea>`);
          } else {
            $$renderer3.push("<!--[!-->");
            Dropdown($$renderer3, {
              label: "",
              show_label: false,
              container: true,
              interactive: true,
              value: line.speaker,
              choices: gradio.props.speakers.map((s) => [s, s]),
              on_change: (val) => update_line(i, "speaker", val)
            });
          }
          $$renderer3.push(`<!--]--></div> <div class="text-column svelte-1p54cvv"><div class="input-container svelte-1p54cvv"><textarea${attr("placeholder", gradio.props.placeholder)}${attr("disabled", disabled, true)} rows="1" class="svelte-1p54cvv">`);
          const $$body_1 = escape_html(line.text);
          if ($$body_1) {
            $$renderer3.push(`${$$body_1}`);
          }
          $$renderer3.push(`</textarea> `);
          {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]--></div></div> `);
          if (gradio.props.max_lines == void 0 || gradio.props.max_lines && i < gradio.props.max_lines - 1) {
            $$renderer3.push("<!--[-->");
            $$renderer3.push(`<div${attr_class("svelte-1p54cvv", void 0, { "action-column": i == 0, "hidden": disabled })}><button class="add-button svelte-1p54cvv" aria-label="Add new line"${attr("disabled", disabled, true)}>`);
            Plus($$renderer3);
            $$renderer3.push(`<!----></button></div>`);
          } else {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]--> <div${attr_class("action-column svelte-1p54cvv", void 0, { "hidden": disabled || i == 0 })}><button class="delete-button svelte-1p54cvv" aria-label="Remove current line"${attr("disabled", disabled, true)}>`);
          Trash($$renderer3);
          $$renderer3.push(`<!----></button></div></div>`);
        }
        $$renderer3.push(`<!--]--></div>`);
      } else {
        $$renderer3.push("<!--[!-->");
        if (checked || gradio.props.ui_mode !== "dialogue") {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<div${attr_class("textarea-container svelte-1p54cvv", void 0, { "loading": is_formatting })}>`);
          {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]--> <textarea data-testid="textbox"${attr("placeholder", gradio.props.placeholder)}${attr("rows", 5)}${attr("disabled", disabled, true)} class="svelte-1p54cvv">`);
          const $$body_2 = escape_html(textbox_value);
          if ($$body_2) {
            $$renderer3.push(`${$$body_2}`);
          }
          $$renderer3.push(`</textarea> `);
          {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]--></div>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]-->`);
      }
      $$renderer3.push(`<!--]--> `);
      if (gradio.props.submit_btn && !disabled) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<div class="submit-container svelte-1p54cvv"><button class="submit-button svelte-1p54cvv"${attr("disabled", disabled, true)}>`);
        if (typeof gradio.props.submit_btn === "string") {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`${escape_html(gradio.props.submit_btn)}`);
        } else {
          $$renderer3.push("<!--[!-->");
          Send($$renderer3);
        }
        $$renderer3.push(`<!--]--></button></div>`);
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--></div>`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    Block($$renderer2, {
      visible: gradio.shared.visible,
      elem_id: gradio.shared.elem_id,
      elem_classes: gradio.shared.elem_classes,
      scale: gradio.shared.scale,
      min_width: gradio.shared.min_width,
      allow_overflow: true,
      padding: gradio.shared.container,
      children: ($$renderer3) => {
        if (gradio.shared.loading_status) {
          $$renderer3.push("<!--[-->");
          Static($$renderer3, spread_props([
            { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
            gradio.shared.loading_status,
            {
              on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
            }
          ]));
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        Dialogue($$renderer3, { gradio });
        $$renderer3.push(`<!---->`);
      },
      $$slots: { default: true }
    });
  });
}

export { Dialogue as BaseDialogue, Index as default };
//# sourceMappingURL=Index46-Ba88O_r1.js.map
