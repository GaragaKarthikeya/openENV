import './async-D55cHugf.js';
import { g as attr_style, i as stringify, b as store_get, e as ensure_array_like, u as unsubscribe_stores, f as attr_class } from './index-u8mz_F03.js';
import { s as sanitize_html_ } from './index50-nVz3eHRG.js';
import { s as spring } from './index3-BSof0MuO.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import { h as html } from './html-CfyvkLET.js';

function ChevronDown($$renderer) {
  $$renderer.push(`<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6L8 10L12 6"></path></svg>`);
}
function Error($$renderer) {
  $$renderer.push(`<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"></path></svg>`);
}
function Info($$renderer) {
  $$renderer.push(`<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path stroke-linecap="round" stroke-linejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z"></path></svg>`);
}
function Success($$renderer) {
  $$renderer.push(`<svg fill="none" stroke="currentColor" viewBox="0 0 24 24" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`);
}
function Warning($$renderer) {
  $$renderer.push(`<svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" stroke-linecap="round" stroke-linejoin="round"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"></path></svg>`);
}
function sanitize(source) {
  return sanitize_html_(source);
}
function ToastContent($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { type, messages = [], expanded = true } = $$props;
    let offset_x = 0;
    const default_messages = {
      error: "An error occurred.",
      warning: "Warning.",
      success: "Success.",
      info: "Info."
    };
    let count = messages.length;
    let first_message = messages[0];
    let type_label = type.charAt(0).toUpperCase() + type.slice(1);
    let has_duration = first_message?.duration !== null;
    let timer_duration = has_duration ? `${first_message.duration}s` : "0s";
    $$renderer2.push(`<div${attr_class(`toast-body ${stringify(type)}`, "svelte-irmu64")} role="status" aria-live="polite" data-testid="toast-body"${attr_style(`transform: translateX(${stringify(offset_x)}px); opacity: ${stringify(1 - Math.abs(offset_x) / 300)};`)}><div class="toast-header svelte-irmu64" role="button" tabindex="0"><div${attr_class(`toast-icon ${stringify(type)}`, "svelte-irmu64")}>`);
    if (type === "warning") {
      $$renderer2.push("<!--[-->");
      Warning($$renderer2);
    } else {
      $$renderer2.push("<!--[!-->");
      if (type === "info") {
        $$renderer2.push("<!--[-->");
        Info($$renderer2);
      } else {
        $$renderer2.push("<!--[!-->");
        if (type === "success") {
          $$renderer2.push("<!--[-->");
          Success($$renderer2);
        } else {
          $$renderer2.push("<!--[!-->");
          if (type === "error") {
            $$renderer2.push("<!--[-->");
            Error($$renderer2);
          } else {
            $$renderer2.push("<!--[!-->");
          }
          $$renderer2.push(`<!--]-->`);
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></div> <div class="toast-title-row svelte-irmu64"><span${attr_class(`toast-title ${stringify(type)}`, "svelte-irmu64")}>${escape_html(type_label)} `);
    if (count > 1) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<span class="toast-count svelte-irmu64">(${escape_html(count)})</span>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></span> <div${attr_class("chevron svelte-irmu64", void 0, { "expanded": expanded, "visible": count > 0 })}>`);
    ChevronDown($$renderer2);
    $$renderer2.push(`<!----></div></div> <button${attr_class(`toast-close ${stringify(type)}`, "svelte-irmu64")} type="button" aria-label="Close" data-testid="toast-close"><span aria-hidden="true">×</span></button></div> `);
    if (expanded) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="toast-messages svelte-irmu64"><!--[-->`);
      const each_array = ensure_array_like(messages);
      for (let i = 0, $$length = each_array.length; i < $$length; i++) {
        let message = each_array[i];
        $$renderer2.push(`<div${attr_class(`toast-message-item ${stringify(type)}`, "svelte-irmu64")}><div${attr_class(`toast-message-text ${stringify(type)}`, "svelte-irmu64")} data-testid="toast-text">${html(sanitize(message.message || default_messages[type]))}</div></div> `);
        if (i < messages.length - 1) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<div class="toast-separator svelte-irmu64"></div>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]--></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> `);
    if (has_duration) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div${attr_class(`timer ${stringify(type)}`, "svelte-irmu64")}${attr_style(`animation-duration: ${stringify(timer_duration)}`)}></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
function Toast($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let { messages = [], on_close } = $$props;
    const top = spring(0, { stiffness: 0.4, damping: 0.5 });
    let grouped_messages = [];
    $$renderer2.push(`<div class="toast-wrap svelte-1qhecvt"${attr_style(`--toast-top: ${stringify(store_get($$store_subs ??= {}, "$top", top))}px;`)}><!--[-->`);
    const each_array = ensure_array_like(grouped_messages);
    for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
      let group = each_array[$$index];
      $$renderer2.push(`<div class="toast-item svelte-1qhecvt">`);
      ToastContent($$renderer2, {
        type: group.type,
        messages: group.messages,
        expanded: group.expanded
      });
      $$renderer2.push(`<!----></div>`);
    }
    $$renderer2.push(`<!--]--></div>`);
    if ($$store_subs) unsubscribe_stores($$store_subs);
  });
}
class LoadingStatus {
  current = {};
  fn_outputs = {};
  fn_inputs = {};
  pending_outputs = /* @__PURE__ */ new Map();
  fn_status = {};
  show_progress = {};
  register(dependency_id, outputs, inputs, show_progress) {
    this.fn_outputs[dependency_id] = outputs;
    this.fn_inputs[dependency_id] = inputs;
    this.show_progress[dependency_id] = show_progress;
  }
  clear(id) {
    if (id in this.current) {
      this.current[id] = {};
    }
  }
  update(args) {
    const updates = this.resolve_args(args);
    updates.forEach(({
      id,
      queue_position,
      queue_size,
      eta,
      status,
      message,
      progress,
      stream_state,
      time_limit,
      type
    }) => {
      this.current[id] = {
        queue: args.queue || false,
        queue_size,
        queue_position,
        eta,
        stream_state,
        message,
        progress: progress || void 0,
        status,
        fn_index: args.fn_index,
        time_limit,
        type,
        show_progress: this.show_progress[args.fn_index]
      };
    });
  }
  set_status(id, status) {
    this.current[id].status = status;
  }
  resolve_args(args) {
    const {
      fn_index,
      status,
      size = void 0,
      position = null,
      eta = null,
      message = null,
      stream_state = null,
      time_limit = null,
      progress_data = null
    } = args;
    const outputs = this.fn_outputs[fn_index];
    const last_status = this.fn_status[fn_index];
    const inputs = this.fn_inputs[fn_index];
    const all_ids = outputs.concat(inputs);
    return all_ids.map((id) => {
      let new_status;
      const pending_count = this.pending_outputs.get(id) || 0;
      if (last_status === "pending" && status !== "pending") {
        let new_count = pending_count - 1;
        this.pending_outputs.set(id, new_count < 0 ? 0 : new_count);
        new_status = new_count > 0 ? "pending" : status;
      } else if (last_status === "pending" && status === "pending") {
        new_status = "pending";
      } else if (last_status !== "pending" && status === "pending") {
        new_status = "pending";
        this.pending_outputs.set(id, pending_count + 1);
      } else {
        new_status = status;
      }
      const type = inputs.includes(id) && stream_state ? "input" : outputs.includes(id) ? "output" : "skip";
      return {
        id,
        queue_position: position,
        queue_size: size,
        eta,
        status: new_status,
        message,
        progress: progress_data,
        stream_state,
        time_limit,
        type
      };
    }).filter((update) => update.type !== "skip");
  }
}

export { LoadingStatus as L, Toast as T };
//# sourceMappingURL=state.svelte-C5Vy1lZM.js.map
