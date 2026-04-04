import { f as fallback } from './async-D55cHugf.js';
import { d as bind_props, f as attr_class, j as clsx, e as ensure_array_like, i as stringify, g as attr_style, a as attr, c as spread_props } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio, S as ShareError, u as uploadToHuggingFace } from './utils.svelte-aYomt3KL.js';
import { I as Image } from './Image-BLyvTcTy.js';
import { c as createEventDispatcher, o as onDestroy } from './index-server-CQz6EZl_.js';
import { I as IconButton } from './IconButton-D82v5nCM.js';
import { C as Check } from './Check-B-uwlXei.js';
import { C as Copy } from './Copy-lixG99xU.js';
import { C as Clear } from './Clear-DH-TDCgr.js';
import { D as Download } from './Download-ByiErn53.js';
import { E as Edit } from './Edit-W_0aHh0i.js';
import { U as Undo } from './Undo-Col2BcUY.js';
import { D as DownloadLink } from './DownloadLink-BMNxSkQ8.js';
import { C as Community, S as ShareButton } from './ShareButton-DnAPtcgb.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { F as File } from './File-2S6P7zIO.js';
import { M as MarkdownCode } from './MarkdownCode-BPzQM9w-.js';
import { T as Trash } from './Trash-DSMnxSso.js';
import { M as Music } from './Music-DcoyPX64.js';
import { B as Block } from './Block-JRVGROlG.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { S as Static } from './index3-BSof0MuO.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './prism-python-uEphcfi3.js';
import './index50-nVz3eHRG.js';
import 'path';
import 'url';
import 'fs';
import './html-CfyvkLET.js';

function Chat($$renderer) {
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--carbon" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path fill="currentColor" d="M17.74 30L16 29l4-7h6a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h9v2H6a4 4 0 0 1-4-4V8a4 4 0 0 1 4-4h20a4 4 0 0 1 4 4v12a4 4 0 0 1-4 4h-4.84Z"></path><path fill="currentColor" d="M8 10h16v2H8zm0 6h10v2H8z"></path></svg>`);
}
function DropdownCircularArrow($$renderer) {
  $$renderer.push(`<svg class="dropdown-arrow svelte-1w1fnc7" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 18 18"><circle cx="9" cy="9" r="8" class="circle svelte-1w1fnc7"></circle><path d="M5 8l4 4 4-4z"></path></svg>`);
}
function Retry($$renderer) {
  $$renderer.push(`<svg width="100%" height="100%" stroke-width="1.5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" color="currentColor"><path d="M19.1679 9C18.0247 6.46819 15.3006 4.5 11.9999 4.5C8.31459 4.5 5.05104 7.44668 4.54932 11" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M16 9H19.4C19.7314 9 20 8.73137 20 8.4V5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M4.88146 15C5.92458 17.5318 8.64874 19.5 12.0494 19.5C15.7347 19.5 18.9983 16.5533 19.5 13" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path><path d="M8.04932 15H4.64932C4.31795 15 4.04932 15.2686 4.04932 15.6V19" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"></path></svg>`);
}
const format_chat_for_sharing = async (chat, url_length_limit = 1800) => {
  let messages_to_share = [...chat];
  let formatted = await format_messages(messages_to_share);
  if (formatted.length > url_length_limit && messages_to_share.length > 2) {
    const first_message = messages_to_share[0];
    const last_message = messages_to_share[messages_to_share.length - 1];
    messages_to_share = [first_message, last_message];
    formatted = await format_messages(messages_to_share);
  }
  if (formatted.length > url_length_limit && messages_to_share.length > 0) {
    const truncated_messages = messages_to_share.map((msg) => {
      if (msg.type === "text") {
        const max_length = Math.floor(url_length_limit / messages_to_share.length) - 20;
        if (msg.content.length > max_length) {
          return {
            ...msg,
            content: msg.content.substring(0, max_length) + "..."
          };
        }
      }
      return msg;
    });
    messages_to_share = truncated_messages;
    formatted = await format_messages(messages_to_share);
  }
  return formatted;
};
const format_messages = async (chat) => {
  let messages = await Promise.all(
    chat.map(async (message) => {
      if (message.role === "system") return "";
      let speaker_emoji = message.role === "user" ? "😃" : "🤖";
      let html_content = "";
      if (message.type === "text") {
        const regexPatterns = {
          audio: /<audio.*?src="(\/file=.*?)"/g,
          video: /<video.*?src="(\/file=.*?)"/g,
          image: /<img.*?src="(\/file=.*?)".*?\/>|!\[.*?\]\((\/file=.*?)\)/g
        };
        html_content = message.content;
        for (let [_, regex] of Object.entries(regexPatterns)) {
          let match;
          while ((match = regex.exec(message.content)) !== null) {
            const fileUrl = match[1] || match[2];
            const newUrl = await uploadToHuggingFace(fileUrl);
            html_content = html_content.replace(fileUrl, newUrl);
          }
        }
      } else {
        if (!message.content.value) return "";
        const url = message.content.component === "video" ? message.content.value?.video.path : message.content.value;
        const file_url = await uploadToHuggingFace(url);
        if (message.content.component === "audio") {
          html_content = `<audio controls src="${file_url}"></audio>`;
        } else if (message.content.component === "video") {
          html_content = file_url;
        } else if (message.content.component === "image") {
          html_content = `<img src="${file_url}" />`;
        }
      }
      return `${speaker_emoji}: ${html_content}`;
    })
  );
  return messages.filter((msg) => msg !== "").join("\n");
};
const redirect_src_url = (src, root) => src.replace('src="/file', `src="${root}file`);
function get_component_for_mime_type(mime_type, file) {
  if (!mime_type) {
    const path = file?.path;
    if (path) {
      const lower_path = path.toLowerCase();
      if (lower_path.endsWith(".glb") || lower_path.endsWith(".gltf") || lower_path.endsWith(".obj") || lower_path.endsWith(".stl") || lower_path.endsWith(".splat") || lower_path.endsWith(".ply")) {
        return "model3d";
      }
    }
    return "file";
  }
  if (mime_type.includes("audio")) return "audio";
  if (mime_type.includes("video")) return "video";
  if (mime_type.includes("image")) return "image";
  if (mime_type.includes("model")) return "model3d";
  return "file";
}
function convert_file_message_to_component_message(message) {
  const _file = Array.isArray(message.file) ? message.file[0] : message.file;
  const component = get_component_for_mime_type(_file?.mime_type, _file);
  return {
    component,
    value: component === "file" ? Array.isArray(message.file) ? message.file : [message.file] : message.file,
    alt_text: message.alt_text,
    constructor_args: {},
    props: {}
  };
}
function normalise_message(message, content, root, i) {
  let normalized;
  if (content.type === "text") {
    normalized = {
      role: message.role,
      metadata: message.metadata,
      content: redirect_src_url(content.text, root),
      type: "text",
      index: i,
      options: message.options
    };
  } else if (content.type === "file") {
    normalized = {
      role: message.role,
      metadata: message.metadata,
      content: convert_file_message_to_component_message(content),
      type: "component",
      index: i,
      options: message.options
    };
  } else {
    normalized = {
      role: message.role,
      metadata: message.metadata,
      content,
      type: "component",
      index: i,
      options: message.options
    };
  }
  return normalized;
}
function normalise_messages(messages, root) {
  if (messages === null) return messages;
  const thought_map = /* @__PURE__ */ new Map();
  return messages.flatMap((message, i) => {
    const normalized = message.content.map(
      (content) => normalise_message(message, content, root, i)
    );
    for (const msg of normalized) {
      const { id, title, parent_id } = message.metadata || {};
      if (parent_id) {
        const parent = thought_map.get(String(parent_id));
        if (parent) {
          const thought = { ...msg, children: [] };
          parent.children.push(thought);
          if (id && title) {
            thought_map.set(String(id), thought);
          }
          return null;
        }
      }
      if (id && title) {
        const thought = { ...msg, children: [] };
        thought_map.set(String(id), thought);
        return thought;
      }
    }
    return normalized;
  }).filter((msg) => msg !== null);
}
function is_component_message(message) {
  return message.type === "component";
}
function is_last_bot_message(messages, all_messages) {
  const is_bot = messages[messages.length - 1].role === "assistant";
  const last_index = messages[messages.length - 1].index;
  const is_last = JSON.stringify(last_index) === JSON.stringify(all_messages[all_messages.length - 1].index);
  return is_last && is_bot;
}
function group_messages(messages, display_consecutive_in_same_bubble = true) {
  const groupedMessages = [];
  let currentGroup = [];
  let currentRole = null;
  for (const message of messages) {
    if (!(message.role === "assistant" || message.role === "user")) {
      continue;
    }
    if (!display_consecutive_in_same_bubble) {
      groupedMessages.push([message]);
      continue;
    }
    if (message.role === currentRole) {
      currentGroup.push(message);
    } else {
      if (currentGroup.length > 0) {
        groupedMessages.push(currentGroup);
      }
      currentGroup = [message];
      currentRole = message.role;
    }
  }
  if (currentGroup.length > 0) {
    groupedMessages.push(currentGroup);
  }
  return groupedMessages;
}
async function load_components(component_names, _components, load_component) {
  for (const component_name of component_names) {
    if (_components[component_name] || component_name === "file") {
      continue;
    }
    const variant = component_name === "dataframe" ? "component" : "base";
    const { component } = load_component(component_name, variant);
    const comp = await component;
    _components[component_name] = comp.default;
  }
  return _components;
}
function get_components_from_messages(messages) {
  if (!messages) return [];
  let components = /* @__PURE__ */ new Set();
  messages.forEach((message) => {
    if (message.type === "component") {
      components.add(message.content.component);
    }
  });
  return Array.from(components);
}
function get_thought_content(msg, depth = 0) {
  let content = "";
  const indent = "  ".repeat(depth);
  if (msg.metadata?.title) {
    content += `${indent}${depth > 0 ? "- " : ""}${msg.metadata.title}
`;
  }
  if (typeof msg.content === "string") {
    content += `${indent}  ${msg.content}
`;
  }
  const thought = msg;
  if (thought.children?.length > 0) {
    content += thought.children.map((child) => get_thought_content(child, depth + 1)).join("");
  }
  return content;
}
function all_text(message) {
  if (Array.isArray(message)) {
    return message.map((m) => {
      if (m.metadata?.title) {
        return get_thought_content(m);
      }
      return m.content;
    }).join("\n");
  }
  if (message.metadata?.title) {
    return get_thought_content(message);
  }
  return message.content;
}
function is_all_text(message) {
  return Array.isArray(message) && message.every((m) => typeof m.content === "string") || !Array.isArray(message) && typeof message.content === "string";
}
function ThumbDownActive($$renderer) {
  $$renderer.push(`<svg width="100%" height="100%" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M11.25 6.61523H9.375V1.36523H11.25V6.61523ZM3.375 1.36523H8.625V6.91636L7.48425 8.62748L7.16737 10.8464C7.14108 11.0248 7.05166 11.1879 6.91535 11.3061C6.77904 11.4242 6.60488 11.4896 6.4245 11.4902H6.375C6.07672 11.4899 5.79075 11.3713 5.57983 11.1604C5.36892 10.9495 5.2503 10.6635 5.25 10.3652V8.11523H2.25C1.85233 8.11474 1.47109 7.95654 1.18989 7.67535C0.908691 7.39415 0.750496 7.01291 0.75 6.61523V3.99023C0.750992 3.29435 1.02787 2.62724 1.51994 2.13517C2.01201 1.64311 2.67911 1.36623 3.375 1.36523Z" fill="currentColor"></path></svg>`);
}
function ThumbDownDefault($$renderer) {
  $$renderer.push(`<svg width="100%" height="100%" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M2.25 8.11523H4.5V10.3652C4.5003 10.6635 4.61892 10.9495 4.82983 11.1604C5.04075 11.3713 5.32672 11.4899 5.625 11.4902H6.42488C6.60519 11.4895 6.77926 11.4241 6.91549 11.3059C7.05172 11.1878 7.14109 11.0248 7.16737 10.8464L7.48425 8.62748L8.82562 6.61523H11.25V1.36523H3.375C2.67911 1.36623 2.01201 1.64311 1.51994 2.13517C1.02787 2.62724 0.750992 3.29435 0.75 3.99023V6.61523C0.750496 7.01291 0.908691 7.39415 1.18989 7.67535C1.47109 7.95654 1.85233 8.11474 2.25 8.11523ZM9 2.11523H10.5V5.86523H9V2.11523ZM1.5 3.99023C1.5006 3.49314 1.69833 3.01657 2.04983 2.66507C2.40133 2.31356 2.8779 2.11583 3.375 2.11523H8.25V6.12661L6.76575 8.35298L6.4245 10.7402H5.625C5.52554 10.7402 5.43016 10.7007 5.35983 10.6304C5.28951 10.5601 5.25 10.4647 5.25 10.3652V7.36523H2.25C2.05118 7.36494 1.86059 7.28582 1.72 7.14524C1.57941 7.00465 1.5003 6.81406 1.5 6.61523V3.99023Z" fill="currentColor"></path></svg>`);
}
function ThumbUpActive($$renderer) {
  $$renderer.push(`<svg width="100%" height="100%" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M0.75 6.24023H2.625V11.4902H0.75V6.24023ZM8.625 11.4902H3.375V5.93911L4.51575 4.22798L4.83263 2.00911C4.85892 1.83065 4.94834 1.66754 5.08465 1.5494C5.22096 1.43125 5.39512 1.36591 5.5755 1.36523H5.625C5.92328 1.36553 6.20925 1.48415 6.42017 1.69507C6.63108 1.90598 6.7497 2.19196 6.75 2.49023V4.74023H9.75C10.1477 4.74073 10.5289 4.89893 10.8101 5.18012C11.0913 5.46132 11.2495 5.84256 11.25 6.24023V8.86523C11.249 9.56112 10.9721 10.2282 10.4801 10.7203C9.98799 11.2124 9.32089 11.4892 8.625 11.4902Z" fill="currentColor"></path></svg>`);
}
function ThumbUpDefault($$renderer) {
  $$renderer.push(`<svg width="100%" height="100%" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9.75 4.74023H7.5V2.49023C7.4997 2.19196 7.38108 1.90598 7.17017 1.69507C6.95925 1.48415 6.67328 1.36553 6.375 1.36523H5.57512C5.39481 1.366 5.22074 1.43138 5.08451 1.54952C4.94828 1.66766 4.85891 1.83072 4.83262 2.00911L4.51575 4.22798L3.17438 6.24023H0.75V11.4902H8.625C9.32089 11.4892 9.98799 11.2124 10.4801 10.7203C10.9721 10.2282 11.249 9.56112 11.25 8.86523V6.24023C11.2495 5.84256 11.0913 5.46132 10.8101 5.18012C10.5289 4.89893 10.1477 4.74073 9.75 4.74023ZM3 10.7402H1.5V6.99023H3V10.7402ZM10.5 8.86523C10.4994 9.36233 10.3017 9.8389 9.95017 10.1904C9.59867 10.5419 9.1221 10.7396 8.625 10.7402H3.75V6.72886L5.23425 4.50248L5.5755 2.11523H6.375C6.47446 2.11523 6.56984 2.15474 6.64017 2.22507C6.71049 2.2954 6.75 2.39078 6.75 2.49023V5.49023H9.75C9.94882 5.49053 10.1394 5.56965 10.28 5.71023C10.4206 5.85082 10.4997 6.04141 10.5 6.24023V8.86523Z" fill="currentColor"></path></svg>`);
}
function Flag($$renderer) {
  $$renderer.push(`<svg id="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="none"><path fill="currentColor" d="M6,30H4V2H28l-5.8,9L28,20H6ZM6,18H24.33L19.8,11l4.53-7H6Z"></path></svg>`);
}
function FlagActive($$renderer) {
  $$renderer.push(`<svg id="icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="none"><path fill="currentColor" d="M4,2H28l-5.8,9L28,20H6v10H4V2z"></path></svg>`);
}
function LikeDislike($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let extra_feedback;
    let i18n = $$props["i18n"];
    let handle_action = $$props["handle_action"];
    let feedback_options = $$props["feedback_options"];
    let selected = fallback($$props["selected"], null);
    function toggleSelection(newSelection) {
      selected = selected === newSelection ? null : newSelection;
      handle_action(selected);
    }
    extra_feedback = feedback_options.filter((option) => option !== "Like" && option !== "Dislike");
    if (feedback_options.includes("Like") || feedback_options.includes("Dislike")) {
      $$renderer2.push("<!--[-->");
      if (feedback_options.includes("Dislike")) {
        $$renderer2.push("<!--[-->");
        IconButton($$renderer2, {
          Icon: selected === "Dislike" ? ThumbDownActive : ThumbDownDefault,
          label: selected === "Dislike" ? "Disliked" : i18n("chatbot.dislike"),
          color: selected === "Dislike" ? "var(--color-accent)" : "var(--block-label-text-color)",
          onclick: () => toggleSelection("Dislike")
        });
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (feedback_options.includes("Like")) {
        $$renderer2.push("<!--[-->");
        IconButton($$renderer2, {
          Icon: selected === "Like" ? ThumbUpActive : ThumbUpDefault,
          label: selected === "Like" ? "Liked" : i18n("chatbot.like"),
          color: selected === "Like" ? "var(--color-accent)" : "var(--block-label-text-color)",
          onclick: () => toggleSelection("Like")
        });
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    if (extra_feedback.length > 0) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="extra-feedback no-border svelte-fvat56">`);
      IconButton($$renderer2, {
        Icon: selected && extra_feedback.includes(selected) ? FlagActive : Flag,
        label: "Feedback",
        color: selected && extra_feedback.includes(selected) ? "var(--color-accent)" : "var(--block-label-text-color)"
      });
      $$renderer2.push(`<!----> <div class="extra-feedback-options svelte-fvat56"><!--[-->`);
      const each_array = ensure_array_like(extra_feedback);
      for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
        let option = each_array[$$index];
        $$renderer2.push(`<button class="extra-feedback-option svelte-fvat56"${attr_style("", { "font-weight": selected === option ? "bold" : "normal" })}>${escape_html(option)}</button>`);
      }
      $$renderer2.push(`<!--]--></div></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { i18n, handle_action, feedback_options, selected });
  });
}
function Copy_1($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let copied = false;
    let value = $$props["value"];
    let watermark = fallback($$props["watermark"], null);
    let i18n = $$props["i18n"];
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
        const text_to_copy = watermark ? `${value}

${watermark}` : value;
        await navigator.clipboard.writeText(text_to_copy);
        copy_feedback();
      } else {
        const textArea = document.createElement("textarea");
        const text_to_copy = watermark ? `${value}

${watermark}` : value;
        textArea.value = text_to_copy;
        textArea.style.position = "absolute";
        textArea.style.left = "-999999px";
        document.body.prepend(textArea);
        textArea.select();
        try {
          document.execCommand("copy");
          copy_feedback();
        } catch (error) {
          console.error(error);
        } finally {
          textArea.remove();
        }
      }
    }
    onDestroy(() => {
      if (timer) clearTimeout(timer);
    });
    IconButton($$renderer2, {
      onclick: handle_copy,
      label: copied ? i18n("chatbot.copied_message") : i18n("chatbot.copy_message"),
      Icon: copied ? Check : Copy
    });
    bind_props($$props, { value, watermark, i18n });
  });
}
function ButtonPanel($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let message_text, show_copy;
    let i18n = $$props["i18n"];
    let likeable = $$props["likeable"];
    let feedback_options = $$props["feedback_options"];
    let show_retry = $$props["show_retry"];
    let show_undo = $$props["show_undo"];
    let show_edit = $$props["show_edit"];
    let in_edit_mode = $$props["in_edit_mode"];
    let show_copy_button = $$props["show_copy_button"];
    let watermark = fallback($$props["watermark"], null);
    let message = $$props["message"];
    let position = $$props["position"];
    let avatar = $$props["avatar"];
    let generating = $$props["generating"];
    let current_feedback = $$props["current_feedback"];
    let file = fallback($$props["file"], null);
    let show_download_button = fallback($$props["show_download_button"], false);
    let show_share_button = fallback($$props["show_share_button"], false);
    let handle_action = $$props["handle_action"];
    let layout = $$props["layout"];
    let dispatch = $$props["dispatch"];
    message_text = is_all_text(message) ? all_text(message) : "";
    show_copy = show_copy_button && message && is_all_text(message);
    if (show_copy || show_retry || show_undo || show_edit || likeable || show_download_button && file?.url || show_share_button && file) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div${attr_class(`message-buttons-${stringify(position)} ${stringify(layout)} message-buttons ${stringify(avatar !== null && "with-avatar")}`, "svelte-704d7y")}>`);
      IconButtonWrapper($$renderer2, {
        top_panel: false,
        children: ($$renderer3) => {
          if (in_edit_mode) {
            $$renderer3.push("<!--[-->");
            IconButton($$renderer3, {
              label: i18n("chatbot.submit"),
              Icon: Check,
              onclick: () => handle_action("edit_submit"),
              disabled: generating
            });
            $$renderer3.push(`<!----> `);
            IconButton($$renderer3, {
              label: i18n("chatbot.cancel"),
              Icon: Clear,
              onclick: () => handle_action("edit_cancel"),
              disabled: generating
            });
            $$renderer3.push(`<!---->`);
          } else {
            $$renderer3.push("<!--[!-->");
            if (show_copy) {
              $$renderer3.push("<!--[-->");
              Copy_1($$renderer3, { value: message_text, watermark, i18n });
            } else {
              $$renderer3.push("<!--[!-->");
            }
            $$renderer3.push(`<!--]--> `);
            if (show_download_button && file?.url) {
              $$renderer3.push("<!--[-->");
              DownloadLink($$renderer3, {
                href: file.is_stream ? file.url?.replace("playlist.m3u8", "playlist-file") : file.url,
                download: file.orig_name || file.path || "file",
                children: ($$renderer4) => {
                  IconButton($$renderer4, { Icon: Download, label: i18n("common.download") });
                },
                $$slots: { default: true }
              });
            } else {
              $$renderer3.push("<!--[!-->");
            }
            $$renderer3.push(`<!--]--> `);
            if (show_share_button && file) {
              $$renderer3.push("<!--[-->");
              ShareButton($$renderer3, {
                i18n,
                formatter: async (value) => {
                  if (!value) return "";
                  let url = await uploadToHuggingFace(value.url);
                  const mime_type = value.mime_type || "";
                  if (mime_type.startsWith("audio/")) {
                    return `<audio controls src="${url}"></audio>`;
                  } else if (mime_type.startsWith("video/")) {
                    return `<video controls src="${url}"></video>`;
                  } else if (mime_type.startsWith("image/")) {
                    return `<img src="${url}" />`;
                  }
                  return "";
                },
                value: file
              });
            } else {
              $$renderer3.push("<!--[!-->");
            }
            $$renderer3.push(`<!--]--> `);
            if (show_retry) {
              $$renderer3.push("<!--[-->");
              IconButton($$renderer3, {
                Icon: Retry,
                label: i18n("chatbot.retry"),
                onclick: () => handle_action("retry"),
                disabled: generating
              });
            } else {
              $$renderer3.push("<!--[!-->");
            }
            $$renderer3.push(`<!--]--> `);
            if (show_undo) {
              $$renderer3.push("<!--[-->");
              IconButton($$renderer3, {
                label: i18n("chatbot.undo"),
                Icon: Undo,
                onclick: () => handle_action("undo"),
                disabled: generating
              });
            } else {
              $$renderer3.push("<!--[!-->");
            }
            $$renderer3.push(`<!--]--> `);
            if (show_edit) {
              $$renderer3.push("<!--[-->");
              IconButton($$renderer3, {
                label: i18n("chatbot.edit"),
                Icon: Edit,
                onclick: () => handle_action("edit"),
                disabled: generating
              });
            } else {
              $$renderer3.push("<!--[!-->");
            }
            $$renderer3.push(`<!--]--> `);
            if (likeable) {
              $$renderer3.push("<!--[-->");
              LikeDislike($$renderer3, {
                handle_action,
                feedback_options,
                selected: current_feedback,
                i18n
              });
            } else {
              $$renderer3.push("<!--[!-->");
            }
            $$renderer3.push(`<!--]-->`);
          }
          $$renderer3.push(`<!--]-->`);
        }
      });
      $$renderer2.push(`<!----></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, {
      i18n,
      likeable,
      feedback_options,
      show_retry,
      show_undo,
      show_edit,
      in_edit_mode,
      show_copy_button,
      watermark,
      message,
      position,
      avatar,
      generating,
      current_feedback,
      file,
      show_download_button,
      show_share_button,
      handle_action,
      layout,
      dispatch
    });
  });
}
function Component($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let type = $$props["type"];
    let components = $$props["components"];
    let value = $$props["value"];
    let target = $$props["target"];
    let theme_mode = $$props["theme_mode"];
    let props = $$props["props"];
    let i18n = $$props["i18n"];
    let upload = $$props["upload"];
    let _fetch = $$props["_fetch"];
    let allow_file_downloads = $$props["allow_file_downloads"];
    let display_icon_button_wrapper_top_corner = fallback($$props["display_icon_button_wrapper_top_corner"], false);
    let image_fullscreen = false;
    if (type === "gallery") {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<!---->`);
      components[type]?.($$renderer2, spread_props([
        props,
        {
          value,
          display_icon_button_wrapper_top_corner: false,
          show_label: props.label ? true : false,
          i18n,
          _fetch,
          allow_preview: false,
          interactive: false,
          mode: "minimal",
          fixed_height: 1
        }
      ]));
      $$renderer2.push(`<!---->`);
    } else {
      $$renderer2.push("<!--[!-->");
      if (type === "dataframe") {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<!---->`);
        components[type]?.($$renderer2, spread_props([
          props,
          {
            value,
            show_label: props.label ? true : false,
            i18n,
            interactive: false,
            line_breaks: props.line_breaks,
            wrap: true,
            root: "",
            gradio: { dispatch: () => {
            }, i18n },
            datatype: props.datatype,
            latex_delimiters: props.latex_delimiters,
            col_count: props.col_count,
            row_count: props.row_count
          }
        ]));
        $$renderer2.push(`<!---->`);
      } else {
        $$renderer2.push("<!--[!-->");
        if (type === "plot") {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<!---->`);
          components[type]?.($$renderer2, spread_props([
            props,
            {
              value,
              target,
              theme_mode,
              on_change: () => {
              },
              bokeh_version: props.bokeh_version,
              caption: props.caption || "",
              show_actions_button: true
            }
          ]));
          $$renderer2.push(`<!---->`);
        } else {
          $$renderer2.push("<!--[!-->");
          if (type === "audio") {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`<!---->`);
            components[type]?.($$renderer2, spread_props([
              props,
              {
                value,
                show_label: props.label ? true : false,
                show_share_button: false,
                i18n,
                waveform_settings: { ...props.waveform_settings, autoplay: props.autoplay },
                show_download_button: false,
                display_icon_button_wrapper_top_corner: false,
                minimal: true
              }
            ]));
            $$renderer2.push(`<!---->`);
          } else {
            $$renderer2.push("<!--[!-->");
            if (type === "video") {
              $$renderer2.push("<!--[-->");
              $$renderer2.push(`<!---->`);
              components[type]?.($$renderer2, spread_props([
                props,
                {
                  autoplay: props.autoplay,
                  value: value.video || value,
                  show_label: props.label ? true : false,
                  show_share_button: false,
                  i18n,
                  upload,
                  display_icon_button_wrapper_top_corner: false,
                  show_download_button: false,
                  children: ($$renderer3) => {
                    $$renderer3.push(`<track kind="captions"/>`);
                  },
                  $$slots: { default: true }
                }
              ]));
              $$renderer2.push(`<!---->`);
            } else {
              $$renderer2.push("<!--[!-->");
              if (type === "image") {
                $$renderer2.push("<!--[-->");
                $$renderer2.push(`<div><!---->`);
                components[type]?.($$renderer2, spread_props([
                  props,
                  {
                    value,
                    show_label: props.label ? true : false,
                    display_icon_button_wrapper_top_corner: false,
                    buttons: ["fullscreen"],
                    fullscreen: image_fullscreen,
                    show_button_background: false,
                    i18n
                  }
                ]));
                $$renderer2.push(`<!----></div>`);
              } else {
                $$renderer2.push("<!--[!-->");
                if (type === "html") {
                  $$renderer2.push("<!--[-->");
                  $$renderer2.push(`<!---->`);
                  components[type]?.($$renderer2, spread_props([props, { props: { value } }]));
                  $$renderer2.push(`<!---->`);
                } else {
                  $$renderer2.push("<!--[!-->");
                  if (type === "model3d") {
                    $$renderer2.push("<!--[-->");
                    $$renderer2.push(`<!---->`);
                    components[type]?.($$renderer2, spread_props([
                      props,
                      {
                        value,
                        clear_color: props.clear_color,
                        display_mode: props.display_mode,
                        zoom_speed: props.zoom_speed,
                        pan_speed: props.pan_speed
                      },
                      props.camera_position !== void 0 && { camera_position: props.camera_position },
                      {
                        has_change_history: true,
                        show_label: props.label ? true : false,
                        root: "",
                        interactive: false,
                        show_share_button: false,
                        gradio: { dispatch: () => {
                        }, i18n },
                        i18n
                      }
                    ]));
                    $$renderer2.push(`<!---->`);
                  } else {
                    $$renderer2.push("<!--[!-->");
                  }
                  $$renderer2.push(`<!--]-->`);
                }
                $$renderer2.push(`<!--]-->`);
              }
              $$renderer2.push(`<!--]-->`);
            }
            $$renderer2.push(`<!--]-->`);
          }
          $$renderer2.push(`<!--]-->`);
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, {
      type,
      components,
      value,
      target,
      theme_mode,
      props,
      i18n,
      upload,
      _fetch,
      allow_file_downloads,
      display_icon_button_wrapper_top_corner
    });
  });
}
function MessageContent($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let latex_delimiters = $$props["latex_delimiters"];
    let sanitize_html = $$props["sanitize_html"];
    let _fetch = $$props["_fetch"];
    let i18n = $$props["i18n"];
    let line_breaks = $$props["line_breaks"];
    let upload = $$props["upload"];
    let target = $$props["target"];
    let theme_mode = $$props["theme_mode"];
    let _components = $$props["_components"];
    let render_markdown = $$props["render_markdown"];
    let scroll2 = $$props["scroll"];
    let allow_file_downloads = $$props["allow_file_downloads"];
    let display_consecutive_in_same_bubble = $$props["display_consecutive_in_same_bubble"];
    let thought_index = $$props["thought_index"];
    let allow_tags = fallback($$props["allow_tags"], false);
    let message = $$props["message"];
    if (message.type === "text") {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="message-content">`);
      MarkdownCode($$renderer2, {
        message: message.content,
        latex_delimiters,
        sanitize_html,
        render_markdown,
        line_breaks,
        allow_tags,
        theme_mode
      });
      $$renderer2.push(`<!----></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
      if (message.type === "component" && message.content.component in _components) {
        $$renderer2.push("<!--[-->");
        Component($$renderer2, {
          target,
          theme_mode,
          props: message.content.props,
          type: message.content.component,
          components: _components,
          value: message.content.value,
          display_icon_button_wrapper_top_corner: thought_index > 0 && display_consecutive_in_same_bubble,
          i18n,
          upload,
          _fetch,
          allow_file_downloads
        });
      } else {
        $$renderer2.push("<!--[!-->");
        if (message.type === "component" && message.content.component === "file") {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<!--[-->`);
          const each_array = ensure_array_like(message.content.value);
          for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
            let file_ = each_array[$$index];
            const file = file_;
            $$renderer2.push(`<div class="file-container svelte-e5gd5s"><div class="file-icon svelte-e5gd5s">`);
            File($$renderer2);
            $$renderer2.push(`<!----></div> <div class="file-info svelte-e5gd5s"><a data-testid="chatbot-file" class="file-link svelte-e5gd5s"${attr("href", file.url)} target="_blank"${attr("download", window.__is_colab__ ? null : file?.orig_name || file?.path.split("/").pop() || "file")}><span class="file-name svelte-e5gd5s">${escape_html(file?.orig_name || file?.path.split("/").pop() || "file")}</span></a> <span class="file-type svelte-e5gd5s">${escape_html((file.orig_name || file.path || "").split(".").pop().toUpperCase())}</span></div></div>`);
          }
          $$renderer2.push(`<!--]-->`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, {
      latex_delimiters,
      sanitize_html,
      _fetch,
      i18n,
      line_breaks,
      upload,
      target,
      theme_mode,
      _components,
      render_markdown,
      scroll: scroll2,
      allow_file_downloads,
      display_consecutive_in_same_bubble,
      thought_index,
      allow_tags,
      message
    });
  });
}
function Thought($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      thought,
      sanitize_html,
      latex_delimiters,
      render_markdown,
      allow_tags
    } = $$props;
    function is_thought_node(msg) {
      return "children" in msg;
    }
    let thought_node = (() => ({
      ...thought,
      children: is_thought_node(thought) ? thought.children : []
    }))();
    let expanded = false;
    $$renderer2.push(`<div class="thought-group svelte-18cn3o3"><div${attr_class("title svelte-18cn3o3", void 0, { "expanded": expanded })}${attr("aria-busy", thought_node.content === "" || thought_node.content === null)} role="button" tabindex="0"><span class="arrow svelte-18cn3o3"${attr_style("", { transform: "rotate(0deg)" })}>`);
    IconButton($$renderer2, { Icon: DropdownCircularArrow });
    $$renderer2.push(`<!----></span> `);
    MarkdownCode($$renderer2, {
      message: thought_node.metadata?.title || "",
      render_markdown,
      latex_delimiters,
      sanitize_html,
      allow_tags: allow_tags || false
    });
    $$renderer2.push(`<!----> `);
    if (thought_node.metadata?.status === "pending") {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<span class="loading-spinner svelte-18cn3o3"></span>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    if (thought_node?.metadata?.log || thought_node?.metadata?.duration) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<span class="duration svelte-18cn3o3">`);
      if (thought_node.metadata.log) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`${escape_html(thought_node.metadata.log)}`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (thought_node.metadata.duration !== void 0) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`(`);
        if (Number.isInteger(thought_node.metadata.duration)) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`${escape_html(thought_node.metadata.duration)}s`);
        } else {
          $$renderer2.push("<!--[!-->");
          if (thought_node.metadata.duration >= 0.1) {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`${escape_html(thought_node.metadata.duration.toFixed(1))}s`);
          } else {
            $$renderer2.push("<!--[!-->");
            $$renderer2.push(`${escape_html((thought_node.metadata.duration * 1e3).toFixed(1))}ms`);
          }
          $$renderer2.push(`<!--]-->`);
        }
        $$renderer2.push(`<!--]-->)`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></span>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div> `);
    {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
function Message($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let value = $$props["value"];
    let avatar_img = $$props["avatar_img"];
    let opposite_avatar_img = fallback($$props["opposite_avatar_img"], null);
    let role = fallback($$props["role"], "user");
    let messages = fallback($$props["messages"], () => [], true);
    let layout = $$props["layout"];
    let render_markdown = $$props["render_markdown"];
    let latex_delimiters = $$props["latex_delimiters"];
    let sanitize_html = $$props["sanitize_html"];
    let selectable = $$props["selectable"];
    let _fetch = $$props["_fetch"];
    let rtl = $$props["rtl"];
    let dispatch = $$props["dispatch"];
    let i18n = $$props["i18n"];
    let line_breaks = $$props["line_breaks"];
    let upload = $$props["upload"];
    let target = $$props["target"];
    let theme_mode = $$props["theme_mode"];
    let _components = $$props["_components"];
    let i = $$props["i"];
    let show_copy_button = $$props["show_copy_button"];
    let generating = $$props["generating"];
    let feedback_options = $$props["feedback_options"];
    let show_like = $$props["show_like"];
    let show_edit = $$props["show_edit"];
    let show_retry = $$props["show_retry"];
    let show_undo = $$props["show_undo"];
    let handle_action = $$props["handle_action"];
    let scroll2 = $$props["scroll"];
    let allow_file_downloads = $$props["allow_file_downloads"];
    let in_edit_mode = $$props["in_edit_mode"];
    let edit_messages = $$props["edit_messages"];
    let display_consecutive_in_same_bubble = $$props["display_consecutive_in_same_bubble"];
    let current_feedback = fallback($$props["current_feedback"], null);
    let allow_tags = fallback($$props["allow_tags"], false);
    let watermark = fallback($$props["watermark"], null);
    let messageElements = [];
    let message_widths = Array(messages.length).fill(160);
    let message_heights = Array(messages.length).fill(0);
    function get_message_label_data(message) {
      if (message.type === "text") {
        return message.content;
      } else if (message.type === "component" && message.content.component === "file") {
        if (Array.isArray(message.content.value)) {
          return `file of extension type: ${message.content.value[0].orig_name?.split(".").pop()}`;
        }
        return `file of extension type: ${message.content.value?.orig_name?.split(".").pop()}` + (message.content.value?.orig_name ?? "");
      }
      return `a component of type ${message.content.component ?? "unknown"}`;
    }
    function get_file(messages2) {
      for (const message of messages2) {
        if (message.type === "component" && (message.content.component === "audio" || message.content.component === "video" || message.content.component === "image" || message.content.component === "file") && message.content.value) {
          return message.content.value;
        }
      }
      return null;
    }
    let button_panel_props;
    if (in_edit_mode && true) {
      const offset = messageElements.length - messages.length;
      for (let idx = offset; idx < messageElements.length; idx++) {
        if (idx >= 0) {
          message_widths[idx - offset] = messageElements[idx]?.clientWidth;
          message_heights[idx - offset] = messageElements[idx]?.clientHeight;
        }
      }
    }
    button_panel_props = {
      handle_action,
      likeable: show_like,
      feedback_options,
      show_retry,
      show_undo,
      show_edit,
      in_edit_mode,
      generating,
      show_copy_button,
      message: messages,
      position: role === "user" ? "right" : "left",
      avatar: avatar_img,
      layout,
      dispatch,
      current_feedback,
      watermark,
      file: get_file(messages),
      show_download_button: allow_file_downloads,
      show_share_button: true
    };
    $$renderer2.push(`<div${attr_class(`message-row ${stringify(layout)} ${stringify(role)}-row`, "svelte-1nr59td", {
      "with_avatar": avatar_img !== null,
      "with_opposite_avatar": opposite_avatar_img !== null
    })}>`);
    if (avatar_img !== null) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="avatar-container svelte-1nr59td">`);
      Image($$renderer2, {
        src: avatar_img?.url});
      $$renderer2.push(`<!----></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> <div${attr_class("flex-wrap svelte-1nr59td", void 0, {
      "role": role,
      "component-wrap": messages[0].type === "component"
    })}><div${attr_class(clsx(display_consecutive_in_same_bubble ? role : ""), "svelte-1nr59td", { "message": display_consecutive_in_same_bubble })}><!--[-->`);
    const each_array = ensure_array_like(messages);
    for (let thought_index = 0, $$length = each_array.length; thought_index < $$length; thought_index++) {
      let message = each_array[thought_index];
      $$renderer2.push(`<div${attr_class(`message ${stringify(!display_consecutive_in_same_bubble ? role : "")}`, "svelte-1nr59td", {
        "panel-full-width": true,
        "message-markdown-disabled": !render_markdown,
        "component": message.type === "component",
        "html": is_component_message(message) && message.content.component === "html",
        "thought": thought_index > 0
      })}>`);
      if (in_edit_mode && message.type === "text") {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<textarea class="edit-textarea svelte-1nr59td" autofocus${attr_style("", {
          width: `max(${message_widths[thought_index]}px, 160px)`,
          "min-height": `${message_heights[thought_index]}px`
        })}>`);
        const $$body = escape_html(edit_messages[thought_index]);
        if ($$body) {
          $$renderer2.push(`${$$body}`);
        }
        $$renderer2.push(`</textarea>`);
      } else {
        $$renderer2.push("<!--[!-->");
        $$renderer2.push(`<div${attr("data-testid", role)}${attr("dir", rtl ? "rtl" : "ltr")}${attr("aria-label", role + "'s message: " + get_message_label_data(message))}${attr_class("svelte-1nr59td", void 0, {
          "latest": i === value.length - 1,
          "message-markdown-disabled": !render_markdown,
          "selectable": selectable
        })}${attr_style("", {
          "user-select": "text",
          cursor: selectable ? "pointer" : "auto",
          "text-align": rtl ? "right" : "left"
        })}>`);
        if (message?.metadata?.title) {
          $$renderer2.push("<!--[-->");
          Thought($$renderer2, {
            thought: message,
            sanitize_html,
            allow_tags,
            latex_delimiters,
            render_markdown
          });
        } else {
          $$renderer2.push("<!--[!-->");
          MessageContent($$renderer2, {
            message,
            sanitize_html,
            allow_tags,
            latex_delimiters,
            render_markdown,
            _components,
            upload,
            thought_index,
            target,
            theme_mode,
            _fetch,
            scroll: scroll2,
            allow_file_downloads,
            display_consecutive_in_same_bubble,
            i18n,
            line_breaks
          });
        }
        $$renderer2.push(`<!--]--></div>`);
      }
      $$renderer2.push(`<!--]--></div> `);
      if (layout === "panel") {
        $$renderer2.push("<!--[-->");
        ButtonPanel($$renderer2, spread_props([button_panel_props, { current_feedback, watermark, i18n }]));
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></div></div></div> `);
    if (layout === "bubble") {
      $$renderer2.push("<!--[-->");
      ButtonPanel($$renderer2, spread_props([button_panel_props, { i18n }]));
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, {
      value,
      avatar_img,
      opposite_avatar_img,
      role,
      messages,
      layout,
      render_markdown,
      latex_delimiters,
      sanitize_html,
      selectable,
      _fetch,
      rtl,
      dispatch,
      i18n,
      line_breaks,
      upload,
      target,
      theme_mode,
      _components,
      i,
      show_copy_button,
      generating,
      feedback_options,
      show_like,
      show_edit,
      show_retry,
      show_undo,
      handle_action,
      scroll: scroll2,
      allow_file_downloads,
      in_edit_mode,
      edit_messages,
      display_consecutive_in_same_bubble,
      current_feedback,
      allow_tags,
      watermark
    });
  });
}
var has = Object.prototype.hasOwnProperty;
function dequal(foo, bar) {
  var ctor, len;
  if (foo === bar) return true;
  if (foo && bar && (ctor = foo.constructor) === bar.constructor) {
    if (ctor === Date) return foo.getTime() === bar.getTime();
    if (ctor === RegExp) return foo.toString() === bar.toString();
    if (ctor === Array) {
      if ((len = foo.length) === bar.length) {
        while (len-- && dequal(foo[len], bar[len])) ;
      }
      return len === -1;
    }
    if (!ctor || typeof foo === "object") {
      len = 0;
      for (ctor in foo) {
        if (has.call(foo, ctor) && ++len && !has.call(bar, ctor)) return false;
        if (!(ctor in bar) || !dequal(foo[ctor], bar[ctor])) return false;
      }
      return Object.keys(bar).length === len;
    }
  }
  return foo !== foo && bar !== bar;
}
function Pending($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let layout = fallback($$props["layout"], "bubble");
    let avatar_images = fallback($$props["avatar_images"], () => [null, null], true);
    $$renderer2.push(`<div class="container svelte-stpvyx">`);
    if (avatar_images[1] !== null) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="avatar-container svelte-stpvyx">`);
      Image($$renderer2, {
        src: avatar_images[1].url
      });
      $$renderer2.push(`<!----></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> <div${attr_class(`message bot pending ${stringify(layout)}`, "svelte-stpvyx", {
      "with_avatar": avatar_images[1] !== null,
      "with_opposite_avatar": avatar_images[0] !== null
    })} role="status" aria-label="Loading response" aria-live="polite"><div class="message-content svelte-stpvyx"><span class="sr-only svelte-stpvyx">Loading content</span> <div class="dots svelte-stpvyx"><div class="dot svelte-stpvyx"></div> <div class="dot svelte-stpvyx"></div> <div class="dot svelte-stpvyx"></div></div></div></div></div>`);
    bind_props($$props, { layout, avatar_images });
  });
}
function Examples($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let examples = fallback($$props["examples"], null);
    let placeholder = fallback($$props["placeholder"], null);
    let latex_delimiters = $$props["latex_delimiters"];
    $$renderer2.push(`<div class="placeholder-content svelte-1rn3hyj" role="complementary">`);
    if (placeholder !== null) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="placeholder svelte-1rn3hyj">`);
      MarkdownCode($$renderer2, { message: placeholder, latex_delimiters });
      $$renderer2.push(`<!----></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    if (examples !== null) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="examples svelte-1rn3hyj" role="list"><!--[-->`);
      const each_array = ensure_array_like(examples);
      for (let i = 0, $$length = each_array.length; i < $$length; i++) {
        let example = each_array[i];
        $$renderer2.push(`<button class="example svelte-1rn3hyj"${attr("aria-label", `Select example ${i + 1}: ${example.display_text || example.text}`)}><div class="example-content svelte-1rn3hyj">`);
        if (example?.icon?.url) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<div class="example-image-container svelte-1rn3hyj">`);
          Image($$renderer2, {
            src: example.icon.url
          });
          $$renderer2.push(`<!----></div>`);
        } else {
          $$renderer2.push("<!--[!-->");
          if (example?.icon?.mime_type === "text") {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`<div class="example-icon svelte-1rn3hyj" aria-hidden="true"><span class="text-icon-aa svelte-1rn3hyj">Aa</span></div>`);
          } else {
            $$renderer2.push("<!--[!-->");
            if (example.files !== void 0 && example.files.length > 0) {
              $$renderer2.push("<!--[-->");
              if (example.files.length > 1) {
                $$renderer2.push("<!--[-->");
                $$renderer2.push(`<div class="example-icons-grid svelte-1rn3hyj" role="group" aria-label="Example attachments"><!--[-->`);
                const each_array_1 = ensure_array_like(example.files.slice(0, 4));
                for (let i2 = 0, $$length2 = each_array_1.length; i2 < $$length2; i2++) {
                  let file = each_array_1[i2];
                  if (file.mime_type?.includes("image")) {
                    $$renderer2.push("<!--[-->");
                    $$renderer2.push(`<div class="example-image-container svelte-1rn3hyj">`);
                    Image($$renderer2, {
                      src: file.url,
                      alt: file.orig_name || `Example image ${i2 + 1}`
                    });
                    $$renderer2.push(`<!----> `);
                    if (i2 === 3 && example.files.length > 4) {
                      $$renderer2.push("<!--[-->");
                      $$renderer2.push(`<div class="image-overlay svelte-1rn3hyj" role="status"${attr("aria-label", `${example.files.length - 4} more files`)}>+${escape_html(example.files.length - 4)}</div>`);
                    } else {
                      $$renderer2.push("<!--[!-->");
                    }
                    $$renderer2.push(`<!--]--></div>`);
                  } else {
                    $$renderer2.push("<!--[!-->");
                    if (file.mime_type?.includes("video")) {
                      $$renderer2.push("<!--[-->");
                      $$renderer2.push(`<div class="example-image-container svelte-1rn3hyj"><video class="example-image"${attr("src", file.url)} aria-hidden="true"></video> `);
                      if (i2 === 3 && example.files.length > 4) {
                        $$renderer2.push("<!--[-->");
                        $$renderer2.push(`<div class="image-overlay svelte-1rn3hyj" role="status"${attr("aria-label", `${example.files.length - 4} more files`)}>+${escape_html(example.files.length - 4)}</div>`);
                      } else {
                        $$renderer2.push("<!--[!-->");
                      }
                      $$renderer2.push(`<!--]--></div>`);
                    } else {
                      $$renderer2.push("<!--[!-->");
                      $$renderer2.push(`<div class="example-icon svelte-1rn3hyj"${attr("aria-label", `File: ${file.orig_name}`)}>`);
                      if (file.mime_type?.includes("audio")) {
                        $$renderer2.push("<!--[-->");
                        Music($$renderer2);
                      } else {
                        $$renderer2.push("<!--[!-->");
                        File($$renderer2);
                      }
                      $$renderer2.push(`<!--]--></div>`);
                    }
                    $$renderer2.push(`<!--]-->`);
                  }
                  $$renderer2.push(`<!--]-->`);
                }
                $$renderer2.push(`<!--]--> `);
                if (example.files.length > 4) {
                  $$renderer2.push("<!--[-->");
                  $$renderer2.push(`<div class="example-icon svelte-1rn3hyj"><div class="file-overlay svelte-1rn3hyj" role="status"${attr("aria-label", `${example.files.length - 4} more files`)}>+${escape_html(example.files.length - 4)}</div></div>`);
                } else {
                  $$renderer2.push("<!--[!-->");
                }
                $$renderer2.push(`<!--]--></div>`);
              } else {
                $$renderer2.push("<!--[!-->");
                if (example.files[0].mime_type?.includes("image")) {
                  $$renderer2.push("<!--[-->");
                  $$renderer2.push(`<div class="example-image-container svelte-1rn3hyj">`);
                  Image($$renderer2, {
                    src: example.files[0].url,
                    alt: example.files[0].orig_name || "Example image"
                  });
                  $$renderer2.push(`<!----></div>`);
                } else {
                  $$renderer2.push("<!--[!-->");
                  if (example.files[0].mime_type?.includes("video")) {
                    $$renderer2.push("<!--[-->");
                    $$renderer2.push(`<div class="example-image-container svelte-1rn3hyj"><video class="example-image"${attr("src", example.files[0].url)} aria-hidden="true"></video></div>`);
                  } else {
                    $$renderer2.push("<!--[!-->");
                    if (example.files[0].mime_type?.includes("audio")) {
                      $$renderer2.push("<!--[-->");
                      $$renderer2.push(`<div class="example-icon svelte-1rn3hyj"${attr("aria-label", `File: ${example.files[0].orig_name}`)}>`);
                      Music($$renderer2);
                      $$renderer2.push(`<!----></div>`);
                    } else {
                      $$renderer2.push("<!--[!-->");
                      $$renderer2.push(`<div class="example-icon svelte-1rn3hyj"${attr("aria-label", `File: ${example.files[0].orig_name}`)}>`);
                      File($$renderer2);
                      $$renderer2.push(`<!----></div>`);
                    }
                    $$renderer2.push(`<!--]-->`);
                  }
                  $$renderer2.push(`<!--]-->`);
                }
                $$renderer2.push(`<!--]-->`);
              }
              $$renderer2.push(`<!--]-->`);
            } else {
              $$renderer2.push("<!--[!-->");
            }
            $$renderer2.push(`<!--]-->`);
          }
          $$renderer2.push(`<!--]-->`);
        }
        $$renderer2.push(`<!--]--> <div class="example-text-content svelte-1rn3hyj"><span class="example-text svelte-1rn3hyj">${escape_html(example.display_text || example.text)}</span></div></div></button>`);
      }
      $$renderer2.push(`<!--]--></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div>`);
    bind_props($$props, { examples, placeholder, latex_delimiters });
  });
}
function CopyAll($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let copied = false;
    let value = $$props["value"];
    let watermark = fallback($$props["watermark"], null);
    let timer;
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
    const copy_conversation = () => {
      if (value) {
        const conversation_value = value.map((message) => {
          if (message.type === "text") {
            return `${message.role}: ${message.content}`;
          }
          return `${message.role}: ${message.content.value.url}`;
        }).join("\n\n");
        const text_to_copy = watermark ? `${conversation_value}

${watermark}` : conversation_value;
        navigator.clipboard.writeText(text_to_copy).catch((err) => {
          console.error("Failed to copy conversation: ", err);
        });
      }
    };
    async function handle_copy() {
      if ("clipboard" in navigator) {
        copy_conversation();
        copy_feedback();
      }
    }
    onDestroy(() => {
      if (timer) clearTimeout(timer);
    });
    IconButton($$renderer2, {
      Icon: copied ? Check : Copy,
      onclick: handle_copy,
      label: copied ? "Copied conversation" : "Copy conversation"
    });
    bind_props($$props, { value, watermark });
  });
}
function ChatBot($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let groupedMessages, options;
    let value = fallback($$props["value"], () => [], true);
    let old_value = null;
    let _fetch = $$props["_fetch"];
    let load_component = $$props["load_component"];
    let allow_file_downloads = $$props["allow_file_downloads"];
    let display_consecutive_in_same_bubble = $$props["display_consecutive_in_same_bubble"];
    let _components = {};
    const is_browser = typeof window !== "undefined";
    async function update_components() {
      _components = await load_components(get_components_from_messages(value), _components, load_component);
    }
    let latex_delimiters = $$props["latex_delimiters"];
    let pending_message = fallback($$props["pending_message"], false);
    let generating = fallback($$props["generating"], false);
    let selectable = fallback($$props["selectable"], false);
    let likeable = fallback($$props["likeable"], false);
    let feedback_options = $$props["feedback_options"];
    let feedback_value = fallback($$props["feedback_value"], null);
    let editable = fallback($$props["editable"], null);
    let show_share_button = fallback($$props["show_share_button"], false);
    let show_copy_all_button = fallback($$props["show_copy_all_button"], false);
    let rtl = fallback($$props["rtl"], false);
    let show_copy_button = fallback($$props["show_copy_button"], false);
    let buttons = fallback($$props["buttons"], null);
    let on_custom_button_click = fallback($$props["on_custom_button_click"], null);
    let avatar_images = fallback($$props["avatar_images"], () => [null, null], true);
    let sanitize_html = fallback($$props["sanitize_html"], true);
    let render_markdown = fallback($$props["render_markdown"], true);
    let line_breaks = fallback($$props["line_breaks"], true);
    let autoscroll = fallback($$props["autoscroll"], true);
    let theme_mode = $$props["theme_mode"];
    let i18n = $$props["i18n"];
    let layout = fallback($$props["layout"], "bubble");
    let placeholder = fallback($$props["placeholder"], null);
    let upload = $$props["upload"];
    let examples = fallback($$props["examples"], null);
    let _retryable = fallback($$props["_retryable"], false);
    let _undoable = fallback($$props["_undoable"], false);
    let like_user_message = fallback($$props["like_user_message"], false);
    let allow_tags = fallback($$props["allow_tags"], false);
    let watermark = fallback($$props["watermark"], null);
    let show_progress = fallback($$props["show_progress"], "full");
    let target = null;
    let edit_index = null;
    let edit_messages = [];
    const dispatch = createEventDispatcher();
    async function scroll_on_value_update() {
      if (!autoscroll) return;
    }
    function handle_action(i, message, selected) {
      if (selected === "undo" || selected === "retry") {
        const val_ = value;
        let last_index = val_.length - 1;
        while (val_[last_index].role === "assistant") {
          last_index--;
        }
        dispatch(selected, {
          index: val_[last_index].index,
          value: val_[last_index].content
        });
      } else if (selected == "edit") {
        edit_index = i;
        edit_messages.push(message.content);
      } else if (selected == "edit_cancel") {
        edit_index = null;
      } else if (selected == "edit_submit") {
        edit_index = null;
        dispatch("edit", {
          index: message.index,
          _dispatch_value: [{ type: "text", text: edit_messages[i].slice() }],
          value: edit_messages[i].slice(),
          previous_value: message.content
        });
      } else {
        let feedback = selected === "Like" ? true : selected === "Dislike" ? false : selected || "";
        if (!groupedMessages) return;
        const message_group = groupedMessages[i];
        const [first] = [message_group[0], message_group[message_group.length - 1]];
        dispatch("like", {
          index: first.index,
          value: message_group.map((m) => m.content),
          liked: feedback
        });
      }
    }
    function get_last_bot_options() {
      if (!value || !groupedMessages || groupedMessages.length === 0) return void 0;
      const last_group = groupedMessages[groupedMessages.length - 1];
      if (last_group[0].role !== "assistant") return void 0;
      return last_group[last_group.length - 1].options;
    }
    get_components_from_messages(value).sort().join(", ");
    update_components();
    if (value || pending_message || _components) {
      scroll_on_value_update();
    }
    {
      if (!dequal(value, old_value)) {
        old_value = value;
      }
    }
    groupedMessages = value && group_messages(value, display_consecutive_in_same_bubble);
    options = value && get_last_bot_options();
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (value !== null && value.length > 0) {
        $$renderer3.push("<!--[-->");
        IconButtonWrapper($$renderer3, {
          buttons,
          on_custom_button_click,
          children: ($$renderer4) => {
            if (show_share_button) {
              $$renderer4.push("<!--[-->");
              IconButton($$renderer4, {
                Icon: Community,
                onclick: async () => {
                  try {
                    const formatted = await format_chat_for_sharing(value);
                    dispatch("share", { description: formatted });
                  } catch (e) {
                    console.error(e);
                    e instanceof ShareError ? e.message : "Share failed.";
                  }
                }
              });
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]--> `);
            IconButton($$renderer4, {
              Icon: Trash,
              onclick: () => dispatch("clear"),
              label: i18n("chatbot.clear")
            });
            $$renderer4.push(`<!----> `);
            if (show_copy_all_button) {
              $$renderer4.push("<!--[-->");
              CopyAll($$renderer4, { value, watermark });
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]-->`);
          }
        });
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> <div${attr_class(clsx(layout === "bubble" ? "bubble-wrap" : "panel-wrap"), "svelte-kpz1")} role="log" aria-label="chatbot conversation" aria-live="polite">`);
      if (value !== null && value.length > 0 && groupedMessages !== null) {
        $$renderer3.push("<!--[-->");
        $$renderer3.push(`<div class="message-wrap svelte-kpz1"><!--[-->`);
        const each_array = ensure_array_like(groupedMessages);
        for (let i = 0, $$length = each_array.length; i < $$length; i++) {
          let messages = each_array[i];
          const role = messages[0].role === "user" ? "user" : "bot";
          const avatar_img = avatar_images[role === "user" ? 0 : 1];
          const opposite_avatar_img = avatar_images[role === "user" ? 0 : 1];
          const feedback_index = groupedMessages.slice(0, i).filter((m) => m[0].role === "assistant").length;
          const current_feedback = role === "bot" && feedback_value && feedback_value[feedback_index] ? feedback_value[feedback_index] : null;
          Message($$renderer3, {
            messages,
            display_consecutive_in_same_bubble,
            opposite_avatar_img,
            avatar_img,
            role,
            layout,
            dispatch,
            i18n,
            _fetch,
            line_breaks,
            theme_mode,
            target,
            upload,
            selectable,
            sanitize_html,
            render_markdown,
            rtl,
            i,
            value,
            latex_delimiters,
            _components,
            generating,
            feedback_options,
            current_feedback,
            allow_tags,
            watermark,
            show_like: role === "user" ? likeable && like_user_message : likeable,
            show_retry: _retryable && is_last_bot_message(messages, value),
            show_undo: _undoable && is_last_bot_message(messages, value),
            show_edit: editable === "all" || editable == "user" && role === "user" && messages.length > 0 && messages[messages.length - 1].type == "text",
            in_edit_mode: edit_index === i,
            show_copy_button,
            handle_action: (selected) => {
              if (selected == "edit") {
                edit_messages.splice(0, edit_messages.length);
              }
              if (selected === "edit" || selected === "edit_submit") {
                messages.forEach((msg, index) => {
                  handle_action(selected === "edit" ? i : index, msg, selected);
                });
              } else {
                handle_action(i, messages[0], selected);
              }
            },
            scroll: is_browser ? scroll : () => {
            },
            allow_file_downloads,
            get edit_messages() {
              return edit_messages;
            },
            set edit_messages($$value) {
              edit_messages = $$value;
              $$settled = false;
            }
          });
          $$renderer3.push(`<!----> `);
          if (show_progress !== "hidden" && generating && messages[messages.length - 1].role === "assistant" && messages[messages.length - 1].metadata?.status === "done") {
            $$renderer3.push("<!--[-->");
            Pending($$renderer3, { layout, avatar_images });
          } else {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]-->`);
        }
        $$renderer3.push(`<!--]--> `);
        if (show_progress !== "hidden" && pending_message) {
          $$renderer3.push("<!--[-->");
          Pending($$renderer3, { layout, avatar_images });
        } else {
          $$renderer3.push("<!--[!-->");
          if (options) {
            $$renderer3.push("<!--[-->");
            $$renderer3.push(`<div class="options svelte-kpz1"><!--[-->`);
            const each_array_1 = ensure_array_like(options);
            for (let index = 0, $$length = each_array_1.length; index < $$length; index++) {
              let option = each_array_1[index];
              $$renderer3.push(`<button class="option svelte-kpz1">${escape_html(option.label || option.value)}</button>`);
            }
            $$renderer3.push(`<!--]--></div>`);
          } else {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]-->`);
        }
        $$renderer3.push(`<!--]--></div>`);
      } else {
        $$renderer3.push("<!--[!-->");
        Examples($$renderer3, { examples, placeholder, latex_delimiters });
      }
      $$renderer3.push(`<!--]--></div> `);
      {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, {
      value,
      _fetch,
      load_component,
      allow_file_downloads,
      display_consecutive_in_same_bubble,
      latex_delimiters,
      pending_message,
      generating,
      selectable,
      likeable,
      feedback_options,
      feedback_value,
      editable,
      show_share_button,
      show_copy_all_button,
      rtl,
      show_copy_button,
      buttons,
      on_custom_button_click,
      avatar_images,
      sanitize_html,
      render_markdown,
      line_breaks,
      autoscroll,
      theme_mode,
      i18n,
      layout,
      placeholder,
      upload,
      examples,
      _retryable,
      _undoable,
      like_user_message,
      allow_tags,
      watermark,
      show_progress
    });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    let _value = normalise_messages(gradio.props.value, gradio.shared.root);
    let show_progress = (() => {
      if (gradio.shared.loading_status.status === "error") {
        return "full";
      }
      return gradio.shared.loading_status.show_progress === "hidden" ? "hidden" : "minimal";
    })();
    Block($$renderer2, {
      elem_id: gradio.shared.elem_id,
      elem_classes: gradio.shared.elem_classes,
      visible: gradio.shared.visible,
      padding: false,
      scale: gradio.shared.scale,
      min_width: gradio.shared.min_width,
      height: gradio.props.height,
      resizable: gradio.props.resizable,
      min_height: gradio.props.min_height,
      max_height: gradio.props.max_height,
      allow_overflow: true,
      flex: true,
      overflow_behavior: "auto",
      rtl: gradio.props.rtl,
      children: ($$renderer3) => {
        if (gradio.shared.loading_status) {
          $$renderer3.push("<!--[-->");
          Static($$renderer3, spread_props([
            { autoscroll: gradio.shared.autoscroll, i18n: gradio.i18n },
            gradio.shared.loading_status,
            {
              show_progress,
              on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
            }
          ]));
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> <div class="wrapper svelte-1wizwbi">`);
        if (gradio.shared.show_label) {
          $$renderer3.push("<!--[-->");
          BlockLabel($$renderer3, {
            show_label: gradio.shared.show_label,
            Icon: Chat,
            float: true,
            label: gradio.shared.label || "Chatbot"
          });
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        ChatBot($$renderer3, {
          i18n: gradio.i18n,
          selectable: gradio.props._selectable,
          likeable: gradio.props.likeable,
          feedback_options: gradio.props.feedback_options,
          feedback_value: gradio.props.feedback_value,
          show_share_button: (gradio.props.buttons ?? ["share"]).some((btn) => typeof btn === "string" && btn === "share"),
          show_copy_all_button: (gradio.props.buttons ?? ["copy_all"]).some((btn) => typeof btn === "string" && btn === "copy_all"),
          value: _value,
          latex_delimiters: gradio.props.latex_delimiters,
          display_consecutive_in_same_bubble: gradio.props.group_consecutive_messages,
          render_markdown: gradio.props.render_markdown,
          theme_mode: gradio.shared.theme_mode,
          editable: gradio.props.editable,
          pending_message: gradio.shared.loading_status?.status === "pending",
          generating: gradio.shared.loading_status?.status === "generating",
          rtl: gradio.props.rtl,
          show_copy_button: (gradio.props.buttons ?? ["copy"]).some((btn) => typeof btn === "string" && btn === "copy"),
          buttons: gradio.props.buttons,
          on_custom_button_click: (id) => {
            gradio.dispatch("custom_button_click", { id });
          },
          like_user_message: gradio.props.like_user_message,
          show_progress: gradio.shared.loading_status?.show_progress || "full",
          avatar_images: (
            //@ts-ignore
            gradio.props.avatar_images
          ),
          sanitize_html: gradio.props.sanitize_html,
          line_breaks: gradio.props.line_breaks,
          autoscroll: gradio.props.autoscroll,
          layout: gradio.props.layout,
          placeholder: gradio.props.placeholder,
          examples: gradio.props.examples,
          _retryable: gradio.props._retryable,
          _undoable: gradio.props._undoable,
          upload: (...args) => gradio.shared.client.upload(...args),
          _fetch: (...args) => gradio.shared.client.fetch(...args),
          load_component: gradio.shared.load_component,
          allow_file_downloads: gradio.props.allow_file_downloads,
          allow_tags: gradio.props.allow_tags,
          watermark: gradio.props.watermark
        });
        $$renderer3.push(`<!----></div>`);
      },
      $$slots: { default: true }
    });
  });
}

export { ChatBot as BaseChatBot, Index as default };
//# sourceMappingURL=Index54-DLt_GW8d.js.map
