import './async-D55cHugf.js';
import { d as bind_props, g as attr_style, e as ensure_array_like, f as attr_class, a as attr, s as slot } from './index-u8mz_F03.js';
import { t as tick } from './index-server-CQz6EZl_.js';
import { U as Upload$1 } from './Upload2-OQ1gRY0J.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { I as IconButton } from './IconButton-D82v5nCM.js';
import { C as Clear } from './Clear-DH-TDCgr.js';
import { F as File } from './File-2S6P7zIO.js';
import { U as Upload } from './Upload-CIQ-D6yx.js';
import './2-BAeILJ1g.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { D as DownloadLink } from './DownloadLink-BMNxSkQ8.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import { h as html } from './html-CfyvkLET.js';

const prettyBytes = (bytes) => {
  let units = ["B", "KB", "MB", "GB", "PB"];
  let i = 0;
  while (bytes > 1024) {
    bytes /= 1024;
    i++;
  }
  let unit = units[i];
  return bytes.toFixed(1) + "&nbsp;" + unit;
};
function FilePreview($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value,
      selectable = false,
      height = void 0,
      i18n,
      allow_reordering = false,
      onselect,
      onchange,
      ondelete,
      ondownload
    } = $$props;
    let dragging_index = null;
    let drop_target_index = null;
    function split_filename(filename) {
      const last_dot = filename.lastIndexOf(".");
      if (last_dot === -1) {
        return [filename, ""];
      }
      return [filename.slice(0, last_dot), filename.slice(last_dot)];
    }
    let normalized_files = (Array.isArray(value) ? value : [value]).map((file) => {
      const [filename_stem, filename_ext] = split_filename(file.orig_name ?? "");
      return { ...file, filename_stem, filename_ext };
    });
    const is_browser = typeof window !== "undefined";
    $$renderer2.push(`<div class="file-preview-holder svelte-al0bnp"${attr_style("", {
      "max-height": height ? typeof height === "number" ? height + "px" : height : "auto"
    })}><table class="file-preview svelte-al0bnp"><tbody class="svelte-al0bnp"><!--[-->`);
    const each_array = ensure_array_like(normalized_files);
    for (let i = 0, $$length = each_array.length; i < $$length; i++) {
      let file = each_array[i];
      $$renderer2.push(`<tr${attr_class("file svelte-al0bnp", void 0, {
        "selectable": selectable,
        "dragging": dragging_index === i,
        "drop-target": drop_target_index === i || i === normalized_files.length - 1 && drop_target_index === normalized_files.length
      })}${attr("data-drop-target", drop_target_index === normalized_files.length && i === normalized_files.length - 1 ? "after" : drop_target_index === i + 1 ? "after" : "before")}${attr("draggable", allow_reordering && normalized_files.length > 1)}><td class="filename svelte-al0bnp"${attr("aria-label", file.orig_name)}>`);
      if (allow_reordering && normalized_files.length > 1) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<span class="drag-handle svelte-al0bnp">⋮⋮</span>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> <span class="stem svelte-al0bnp">${escape_html(file.filename_stem)}</span> <span class="ext svelte-al0bnp">${escape_html(file.filename_ext)}</span></td><td class="download svelte-al0bnp">`);
      if (file.url) {
        $$renderer2.push("<!--[-->");
        DownloadLink($$renderer2, {
          href: file.url,
          download: is_browser && window.__is_colab__ ? null : file.orig_name,
          children: ($$renderer3) => {
            $$renderer3.push(`${html(file.size != null ? prettyBytes(file.size) : "(size unknown)")} ⇣`);
          },
          $$slots: { default: true }
        });
      } else {
        $$renderer2.push("<!--[!-->");
        $$renderer2.push(`${escape_html(i18n("file.uploading"))}`);
      }
      $$renderer2.push(`<!--]--></td>`);
      if (normalized_files.length > 1) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<td class="svelte-al0bnp"><button class="label-clear-button svelte-al0bnp" aria-label="Remove this file">×</button></td>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></tr>`);
    }
    $$renderer2.push(`<!--]--></tbody></table></div>`);
  });
}
function FileUpload($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      value = void 0,
      label,
      show_label = true,
      file_count = "single",
      file_types = null,
      selectable = false,
      root,
      height = void 0,
      i18n,
      max_file_size = null,
      upload,
      stream_handler,
      uploading = false,
      allow_reordering = false,
      upload_promise = void 0,
      buttons = null,
      on_custom_button_click = null,
      onchange,
      onclear,
      ondrag,
      onupload,
      onerror,
      ondelete,
      onselect
    } = $$props;
    async function handle_upload(detail) {
      if (Array.isArray(value)) {
        value = [...value, ...Array.isArray(detail) ? detail : [detail]];
      } else if (value) {
        value = [value, ...Array.isArray(detail) ? detail : [detail]];
      } else {
        value = detail;
      }
      await tick();
      onchange?.(value);
      onupload?.(value);
    }
    function handle_clear() {
      value = null;
      onchange?.(null);
      onclear?.();
    }
    let dragging = false;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (show_label && buttons && buttons.length > 0) {
        $$renderer3.push("<!--[-->");
        IconButtonWrapper($$renderer3, { buttons, on_custom_button_click });
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      BlockLabel($$renderer3, {
        show_label,
        Icon: File,
        float: !value,
        label: label || "File"
      });
      $$renderer3.push(`<!----> `);
      if (value && (Array.isArray(value) ? value.length > 0 : true)) {
        $$renderer3.push("<!--[-->");
        IconButtonWrapper($$renderer3, {
          buttons: buttons || [],
          on_custom_button_click,
          children: ($$renderer4) => {
            if (!(file_count === "single" && (Array.isArray(value) ? value.length > 0 : value !== null))) {
              $$renderer4.push("<!--[-->");
              IconButton($$renderer4, {
                Icon: Upload,
                label: i18n("common.upload"),
                children: ($$renderer5) => {
                  Upload$1($$renderer5, {
                    icon_upload: true,
                    onload: handle_upload,
                    filetype: file_types,
                    file_count,
                    max_file_size,
                    root,
                    onerror,
                    stream_handler,
                    upload,
                    get upload_promise() {
                      return upload_promise;
                    },
                    set upload_promise($$value) {
                      upload_promise = $$value;
                      $$settled = false;
                    },
                    get dragging() {
                      return dragging;
                    },
                    set dragging($$value) {
                      dragging = $$value;
                      $$settled = false;
                    },
                    get uploading() {
                      return uploading;
                    },
                    set uploading($$value) {
                      uploading = $$value;
                      $$settled = false;
                    }
                  });
                }
              });
            } else {
              $$renderer4.push("<!--[!-->");
            }
            $$renderer4.push(`<!--]--> `);
            IconButton($$renderer4, {
              Icon: Clear,
              label: i18n("common.clear"),
              onclick: (event) => {
                event.stopPropagation();
                handle_clear();
              }
            });
            $$renderer4.push(`<!---->`);
          }
        });
        $$renderer3.push(`<!----> `);
        FilePreview($$renderer3, {
          i18n,
          onselect,
          selectable,
          value,
          height,
          onchange,
          ondelete,
          allow_reordering
        });
        $$renderer3.push(`<!---->`);
      } else {
        $$renderer3.push("<!--[!-->");
        Upload$1($$renderer3, {
          onload: handle_upload,
          filetype: file_types,
          file_count,
          max_file_size,
          root,
          onerror,
          stream_handler,
          upload,
          height,
          get upload_promise() {
            return upload_promise;
          },
          set upload_promise($$value) {
            upload_promise = $$value;
            $$settled = false;
          },
          get dragging() {
            return dragging;
          },
          set dragging($$value) {
            dragging = $$value;
            $$settled = false;
          },
          get uploading() {
            return uploading;
          },
          set uploading($$value) {
            uploading = $$value;
            $$settled = false;
          },
          children: ($$renderer4) => {
            $$renderer4.push(`<!--[-->`);
            slot($$renderer4, $$props, "default", {});
            $$renderer4.push(`<!--]-->`);
          },
          $$slots: { default: true }
        });
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { value, uploading, upload_promise });
  });
}

export { FilePreview as F, FileUpload as a };
//# sourceMappingURL=FileUpload-Bz0iVHB2.js.map
