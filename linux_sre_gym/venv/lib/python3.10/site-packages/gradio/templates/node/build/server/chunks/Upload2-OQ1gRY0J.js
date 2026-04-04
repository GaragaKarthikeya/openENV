import './async-D55cHugf.js';
import { a as attr, f as attr_class, g as attr_style, d as bind_props } from './index-u8mz_F03.js';
import { o as onDestroy, t as tick } from './index-server-CQz6EZl_.js';
import { a as prepare_files } from './2-BAeILJ1g.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';

/* empty css                                     */
function UploadProgress($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { upload_id, root, files, stream_handler, ondone } = $$props;
    let progress = false;
    let file_to_display = files_with_progress[0];
    let files_with_progress = files.map((file) => {
      return { ...file, progress: 0 };
    });
    function getProgress(file) {
      return file.progress * 100 / (file.size || 0) || 0;
    }
    onDestroy(() => {
    });
    $$renderer2.push(`<div${attr_class("wrap svelte-ua961l", void 0, { "progress": progress })}><span class="uploading svelte-ua961l">Uploading ${escape_html(files_with_progress.length)}
		${escape_html(files_with_progress.length > 1 ? "files" : "file")}...</span> `);
    if (file_to_display) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="file svelte-ua961l"><span><div class="progress-bar svelte-ua961l"><progress style="visibility:hidden;height:0;width:0;"${attr("value", getProgress(file_to_display))} max="100" class="svelte-ua961l">${escape_html(getProgress(file_to_display))}</progress></div></span> <span class="file-name svelte-ua961l">${escape_html(file_to_display.orig_name)}</span></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div>`);
  });
}
function Upload($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      filetype = null,
      dragging = false,
      boundedheight = true,
      center = true,
      flex = true,
      file_count = "single",
      disable_click = false,
      root,
      hidden = false,
      format = "file",
      uploading = false,
      show_progress = true,
      max_file_size = null,
      upload,
      stream_handler,
      icon_upload = false,
      height = void 0,
      aria_label = void 0,
      upload_promise = void 0,
      onload,
      onerror,
      children
    } = $$props;
    function open_upload() {
    }
    let upload_id = "";
    let file_data;
    let use_post_upload_validation = null;
    const get_ios = () => {
      if (typeof navigator !== "undefined") {
        const userAgent = navigator.userAgent.toLowerCase();
        return userAgent.indexOf("iphone") > -1 || userAgent.indexOf("ipad") > -1;
      }
      return false;
    };
    let ios = get_ios();
    const validFileTypes = ["image", "video", "audio", "text", "file"];
    const process_file_type = (type) => {
      if (ios && type.startsWith(".")) {
        use_post_upload_validation = true;
        return type;
      }
      if (ios && type.includes("file/*")) {
        return "*";
      }
      if (type.startsWith(".") || type.endsWith("/*")) {
        return type;
      }
      if (validFileTypes.includes(type)) {
        return type + "/*";
      }
      return "." + type;
    };
    function paste_clipboard() {
      navigator.clipboard.read().then(async (items) => {
        for (let i = 0; i < items.length; i++) {
          const type = items[i].types.find((t) => t.startsWith("image/"));
          if (type) {
            items[i].getType(type).then(async (blob) => {
              const file = new File([blob], `clipboard.${type.replace("image/", "")}`);
              await load_files([file]);
            });
            break;
          }
        }
      });
    }
    function open_file_upload() {
    }
    async function handle_upload(file_data2, _upload_id) {
      if (!_upload_id) {
        upload_id = Math.random().toString(36).substring(2, 15);
      } else {
        upload_id = _upload_id;
      }
      await tick();
      uploading = true;
      upload_promise = new Promise(async (resolve) => {
        try {
          const _file_data = await upload(file_data2, root, upload_id, max_file_size ?? Infinity);
          onload?.(file_count === "single" ? _file_data?.[0] : _file_data);
          resolve(_file_data || []);
          uploading = false;
        } catch (e) {
          onerror?.(e.message);
          uploading = false;
          resolve([]);
        }
      });
      return upload_promise;
    }
    async function load_files(files, upload_id2) {
      if (!files.length) {
        return;
      }
      let _files = files.map((f) => new File([f], f instanceof File ? f.name : "file", { type: f.type }));
      if (ios && use_post_upload_validation) {
        _files = _files.filter((file) => {
          if (is_valid_file(file)) {
            return true;
          }
          onerror?.(`Invalid file type: ${file.name}. Only ${filetype} allowed.`);
          return false;
        });
        if (_files.length === 0) {
          return [];
        }
      }
      file_data = await prepare_files(_files);
      return await handle_upload(file_data, upload_id2);
    }
    function is_valid_file(file) {
      if (!filetype) return true;
      const allowed_types = Array.isArray(filetype) ? filetype : [filetype];
      return allowed_types.some((type) => {
        const processed_type = process_file_type(type);
        if (processed_type.startsWith(".")) {
          return file.name.toLowerCase().endsWith(processed_type.toLowerCase());
        }
        if (processed_type === "*") {
          return true;
        }
        if (processed_type.endsWith("/*")) {
          const [category] = processed_type.split("/");
          return file.type.startsWith(category + "/");
        }
        return file.type === processed_type;
      });
    }
    async function load_files_from_drop(e) {
      dragging = false;
      if (!e.dataTransfer?.files) return;
      const files_to_load = Array.from(e.dataTransfer.files).filter(is_valid_file);
      if (format != "blob") {
        await load_files(files_to_load);
      } else {
        if (file_count === "single") {
          onload?.(files_to_load[0]);
          return;
        }
        onload?.(files_to_load);
      }
    }
    if (filetype === "clipboard") {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<button${attr("tabindex", hidden ? -1 : 0)}${attr("aria-label", aria_label || "Paste from clipboard")}${attr_class("svelte-8prmba", void 0, {
        "hidden": hidden,
        "center": center,
        "boundedheight": boundedheight,
        "flex": flex,
        "icon-mode": icon_upload
      })}${attr_style("", {
        height: icon_upload ? "" : height ? typeof height === "number" ? height + "px" : height : "100%"
      })}>`);
      if (children) {
        $$renderer2.push("<!--[-->");
        children($$renderer2);
        $$renderer2.push(`<!---->`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></button>`);
    } else {
      $$renderer2.push("<!--[!-->");
      if (uploading && show_progress) {
        $$renderer2.push("<!--[-->");
        if (!hidden) {
          $$renderer2.push("<!--[-->");
          UploadProgress($$renderer2, { root, upload_id, files: file_data, stream_handler });
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]-->`);
      } else {
        $$renderer2.push("<!--[!-->");
        $$renderer2.push(`<button${attr("tabindex", hidden ? -1 : 0)}${attr("aria-label", aria_label || "Click to upload or drop files")} aria-dropeffect="copy"${attr_class("svelte-8prmba", void 0, {
          "hidden": hidden,
          "center": center,
          "boundedheight": boundedheight,
          "flex": flex,
          "disable_click": disable_click,
          "icon-mode": icon_upload
        })}${attr_style("", {
          height: icon_upload ? "" : height ? typeof height === "number" ? height + "px" : height : "100%"
        })}>`);
        if (children) {
          $$renderer2.push("<!--[-->");
          children($$renderer2);
          $$renderer2.push(`<!---->`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]--></button>`);
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, {
      dragging,
      uploading,
      upload_promise,
      open_upload,
      paste_clipboard,
      open_file_upload,
      load_files,
      load_files_from_drop
    });
  });
}

export { Upload as U, UploadProgress as a };
//# sourceMappingURL=Upload2-OQ1gRY0J.js.map
