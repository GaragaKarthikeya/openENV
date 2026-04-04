import './async-D55cHugf.js';
import { d as bind_props, f as attr_class, g as attr_style, i as stringify, e as ensure_array_like, a as attr } from './index-u8mz_F03.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { I as IconButton } from './IconButton-D82v5nCM.js';
import { E as Empty } from './Empty-BHvMDNv5.js';
import { S as ShareButton } from './ShareButton-DnAPtcgb.js';
import { C as Clear } from './Clear-DH-TDCgr.js';
import { D as Download } from './Download-ByiErn53.js';
import { I as Image } from './Image2-BLbvQKZw.js';
import { U as Upload, I as ImagePaste } from './Upload-CIQ-D6yx.js';
import { P as Play } from './Play-BlZWjD1q.js';
import { V as Video$1 } from './Video-_1zl9-Cr.js';
import { W as Webcam } from './Webcam-iV1BaoQQ.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { F as FullscreenButton } from './FullscreenButton-DVAVExlR.js';
import { F as FileData } from './2-BAeILJ1g.js';
import { U as Upload$1 } from './Upload2-OQ1gRY0J.js';
import { M as ModifyUpload } from './ModifyUpload-D_fhEqLO.js';
import { I as Image$1 } from './Image-BLyvTcTy.js';
import { V as Video } from './Video2-De5bYQY6.js';
import { u as uploadToHuggingFace } from './utils.svelte-aYomt3KL.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index-server-CQz6EZl_.js';
import './Maximize-B77VDSzq.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './DownloadLink-BMNxSkQ8.js';
import './Edit-W_0aHh0i.js';
import './Undo-Col2BcUY.js';
import './clone-Yk88IHKV.js';

async function format_gallery_for_sharing(value) {
  if (!value) return "";
  let urls = await Promise.all(
    value.map(async ([image, _]) => {
      if (image === null || !image.url) return "";
      return await uploadToHuggingFace(image.url);
    })
  );
  return `<div style="display: flex; flex-wrap: wrap; gap: 16px">${urls.map((url) => `<img src="${url}" style="height: 400px" />`).join("")}</div>`;
}
async function handle_save(img_blob, upload, filename = "uploaded_file") {
  const ext = img_blob.type.split("/")[1] || "png";
  const f_ = new File([img_blob], `${filename}.${ext}`);
  const files = [
    new FileData({
      path: f_.name,
      orig_name: f_.name,
      blob: f_,
      size: f_.size,
      mime_type: f_.type,
      is_stream: false
    })
  ];
  return await upload(...files);
}
function Gallery($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      show_label = true,
      label,
      value = void 0,
      columns = [2],
      rows = void 0,
      height = "auto",
      preview,
      allow_preview = true,
      object_fit = "cover",
      show_share_button = false,
      show_download_button = false,
      i18n,
      selected_index = void 0,
      interactive,
      _fetch,
      show_fullscreen_button = true,
      fullscreen = false,
      root = "",
      file_types = ["image", "video"],
      sources,
      max_file_size = null,
      upload,
      stream_handler,
      fit_columns = true,
      buttons = null,
      oncustom_button_click = null,
      onpreview_open = () => {
      },
      onchange = () => {
      },
      onclear = () => {
      },
      onselect = (data) => {
      },
      onshare = (data) => {
      },
      onerror = (error) => {
      },
      onpreview_close = () => {
      },
      onfullscreen = (data) => {
      },
      ondelete = () => {
      },
      onupload = () => {
      },
      onsource_change = () => {
      }
    } = $$props;
    let upload_promise = null;
    let mode = "normal";
    let display_icon_button_wrapper_top_corner = false;
    let is_full_screen = false;
    let resolved_value = (() => value == null ? null : value.map((data) => {
      if ("video" in data) {
        return { video: data.video, caption: data.caption };
      } else if ("image" in data) {
        return { image: data.image, caption: data.caption };
      }
      return {};
    }))();
    function resolve_effective_columns(resolved_value2, columns2, fit_columns2) {
      if (resolved_value2 && columns2 && fit_columns2) {
        const item_count = resolved_value2.length;
        if (Array.isArray(columns2)) {
          return columns2.map((col) => Math.min(col, item_count));
        } else {
          return Math.min(columns2, item_count);
        }
      } else {
        return columns2;
      }
    }
    let effective_columns = (() => resolve_effective_columns(resolved_value, columns, fit_columns))();
    if (selected_index == null && preview && value?.length) {
      selected_index = 0;
    }
    (() => ((selected_index ?? 0) + (resolved_value?.length ?? 0) - 1) % (resolved_value?.length ?? 0))();
    (() => ((selected_index ?? 0) + 1) % (resolved_value?.length ?? 0))();
    async function download(file_url, name) {
      let response;
      try {
        response = await _fetch(file_url);
      } catch (error) {
        if (error instanceof TypeError) {
          window.open(file_url, "_blank", "noreferrer");
          return;
        }
        throw error;
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = name;
      link.click();
      URL.revokeObjectURL(url);
    }
    let selected_media = (() => selected_index != null && resolved_value != null ? resolved_value[selected_index] : null)();
    let uploading = false;
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      if (show_label) {
        $$renderer3.push("<!--[-->");
        BlockLabel($$renderer3, { show_label, Icon: Image, label: label || "Gallery" });
      } else {
        $$renderer3.push("<!--[!-->");
      }
      $$renderer3.push(`<!--]--> `);
      if (value == null || resolved_value == null || resolved_value.length === 0) {
        $$renderer3.push("<!--[-->");
        Empty($$renderer3, {
          unpadded_box: true,
          size: "large",
          children: ($$renderer4) => {
            Image($$renderer4);
          },
          $$slots: { default: true }
        });
      } else {
        $$renderer3.push("<!--[!-->");
        $$renderer3.push(`<div class="gallery-container svelte-7anmrz">`);
        if (selected_media && allow_preview) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<span${attr_class("preview svelte-7anmrz", void 0, { "minimal": mode === "minimal" })}>`);
          IconButtonWrapper($$renderer3, {
            display_top_corner: display_icon_button_wrapper_top_corner,
            buttons,
            on_custom_button_click: oncustom_button_click,
            children: ($$renderer4) => {
              if (show_download_button) {
                $$renderer4.push("<!--[-->");
                IconButton($$renderer4, {
                  Icon: Download,
                  label: i18n("common.download"),
                  onclick: () => {
                    const image = "image" in selected_media ? selected_media?.image : selected_media?.video;
                    if (image == null) {
                      return;
                    }
                    const { url, orig_name } = image;
                    if (url) {
                      download(url, orig_name ?? "image");
                    }
                  }
                });
              } else {
                $$renderer4.push("<!--[!-->");
              }
              $$renderer4.push(`<!--]--> `);
              if (show_fullscreen_button) {
                $$renderer4.push("<!--[-->");
                FullscreenButton($$renderer4, {
                  fullscreen,
                  onclick: (is_fullscreen) => onfullscreen({ detail: is_fullscreen })
                });
              } else {
                $$renderer4.push("<!--[!-->");
              }
              $$renderer4.push(`<!--]--> `);
              if (show_share_button) {
                $$renderer4.push("<!--[-->");
                $$renderer4.push(`<div class="icon-button">`);
                ShareButton($$renderer4, {
                  i18n,
                  value: resolved_value,
                  formatter: format_gallery_for_sharing
                });
                $$renderer4.push(`<!----></div>`);
              } else {
                $$renderer4.push("<!--[!-->");
              }
              $$renderer4.push(`<!--]--> `);
              {
                $$renderer4.push("<!--[-->");
                IconButton($$renderer4, {
                  Icon: Clear,
                  label: "Close",
                  onclick: () => {
                    selected_index = null;
                    onpreview_close();
                  }
                });
              }
              $$renderer4.push(`<!--]-->`);
            }
          });
          $$renderer3.push(`<!----> <button class="media-button svelte-7anmrz"${attr_style(`height: calc(100% - ${stringify(selected_media.caption ? "80px" : "60px")})`)} aria-label="detailed view of selected image">`);
          if ("image" in selected_media) {
            $$renderer3.push("<!--[-->");
            Image$1($$renderer3, {
              restProps: {
                alt: selected_media.caption || "",
                title: selected_media.caption || null,
                class: selected_media.caption && "with-caption",
                loading: "lazy"
              },
              src: selected_media.image.url,
              data_testid: "detailed-image"
            });
          } else {
            $$renderer3.push("<!--[!-->");
            Video($$renderer3, {
              src: selected_media.video.url,
              "data-testid": "detailed-video",
              alt: selected_media.caption || "",
              loading: "lazy",
              loop: false,
              is_stream: false,
              muted: false,
              controls: true
            });
          }
          $$renderer3.push(`<!--]--></button> `);
          if (selected_media?.caption) {
            $$renderer3.push("<!--[-->");
            $$renderer3.push(`<caption class="caption svelte-7anmrz">${escape_html(selected_media.caption)}</caption>`);
          } else {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]--> <div class="thumbnails scroll-hide svelte-7anmrz" data-testid="container_el"${attr_style(`justify-content: ${stringify("center")};`)}><!--[-->`);
          const each_array = ensure_array_like(resolved_value);
          for (let i = 0, $$length = each_array.length; i < $$length; i++) {
            let media = each_array[i];
            $$renderer3.push(`<button${attr_class("thumbnail-item thumbnail-small svelte-7anmrz", void 0, { "selected": selected_index === i && mode !== "minimal" })}${attr("aria-label", "Thumbnail " + (i + 1) + " of " + resolved_value.length)}>`);
            if ("image" in media) {
              $$renderer3.push("<!--[-->");
              Image$1($$renderer3, {
                src: media.image.url,
                restProps: {
                  title: media.caption || null,
                  alt: "",
                  class: "with-caption",
                  loading: "lazy"
                },
                data_testid: `thumbnail ${i + 1}`
              });
            } else {
              $$renderer3.push("<!--[!-->");
              Play($$renderer3);
              $$renderer3.push(`<!----> `);
              Video($$renderer3, {
                src: media.video.url,
                title: media.caption || null,
                is_stream: false,
                "data-testid": "thumbnail " + (i + 1),
                alt: "",
                loading: "lazy",
                loop: false
              });
              $$renderer3.push(`<!---->`);
            }
            $$renderer3.push(`<!--]--></button>`);
          }
          $$renderer3.push(`<!--]--></div></span>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> <div${attr_class("grid-wrap svelte-7anmrz", void 0, {
          "minimal": mode === "minimal",
          "fixed-height": !height || height == "auto",
          "hidden": is_full_screen
        })}${attr_style("", {
          height: height !== "auto" ? typeof height === "number" ? height + "px" : height : null
        })}>`);
        if (interactive && selected_index === null) {
          $$renderer3.push("<!--[-->");
          ModifyUpload($$renderer3, {
            i18n,
            onclear: () => {
              value = [];
              onsource_change("upload");
              onclear();
            },
            children: ($$renderer4) => {
              if (upload && stream_handler) {
                $$renderer4.push("<!--[-->");
                IconButton($$renderer4, {
                  Icon: Upload,
                  label: i18n("upload_text.click_to_upload"),
                  children: ($$renderer5) => {
                    Upload$1($$renderer5, {
                      icon_upload: true,
                      onload: (e) => onupload(e),
                      filetype: file_types,
                      file_count: "multiple",
                      max_file_size,
                      root,
                      onerror: (e) => onerror(e),
                      stream_handler,
                      upload,
                      get upload_promise() {
                        return upload_promise;
                      },
                      set upload_promise($$value) {
                        upload_promise = $$value;
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
              if (sources.includes("webcam")) {
                $$renderer4.push("<!--[-->");
                IconButton($$renderer4, {
                  Icon: Webcam,
                  label: i18n("common.webcam"),
                  onclick: () => {
                    onsource_change("webcam");
                  }
                });
              } else {
                $$renderer4.push("<!--[!-->");
              }
              $$renderer4.push(`<!--]--> `);
              if (sources.includes("webcam-video")) {
                $$renderer4.push("<!--[-->");
                IconButton($$renderer4, {
                  Icon: Video$1,
                  label: i18n("common.video"),
                  onclick: () => {
                    onsource_change("webcam-video");
                  }
                });
              } else {
                $$renderer4.push("<!--[!-->");
              }
              $$renderer4.push(`<!--]--> `);
              if (sources.includes("clipboard")) {
                $$renderer4.push("<!--[-->");
                IconButton($$renderer4, {
                  label: i18n("upload_text.paste_clipboard"),
                  onclick: () => {
                    onsource_change("clipboard");
                  },
                  Icon: ImagePaste
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
        $$renderer3.push(`<!--]--> <div${attr_class("grid-container svelte-7anmrz", void 0, { "pt-6": show_label })}${attr_style(`--grid-cols:${stringify(effective_columns)}; --grid-rows:${stringify(rows)}; --object-fit: ${stringify(object_fit)};`)}><!--[-->`);
        const each_array_1 = ensure_array_like(resolved_value);
        for (let i = 0, $$length = each_array_1.length; i < $$length; i++) {
          let entry = each_array_1[i];
          $$renderer3.push(`<div class="gallery-item svelte-7anmrz"><button${attr_class("thumbnail-item thumbnail-lg svelte-7anmrz", void 0, { "selected": selected_index === i })}${attr("aria-label", "Thumbnail " + (i + 1) + " of " + resolved_value.length)}>`);
          if ("image" in entry) {
            $$renderer3.push("<!--[-->");
            Image$1($$renderer3, {
              alt: entry.caption || "",
              src: typeof entry.image === "string" ? entry.image : entry.image.url
            });
          } else {
            $$renderer3.push("<!--[!-->");
            Play($$renderer3);
            $$renderer3.push(`<!----> `);
            Video($$renderer3, {
              src: entry.video.url,
              title: entry.caption || null,
              is_stream: false,
              "data-testid": "thumbnail " + (i + 1),
              alt: "",
              loading: "lazy",
              loop: false
            });
            $$renderer3.push(`<!---->`);
          }
          $$renderer3.push(`<!--]--> `);
          if (entry.caption) {
            $$renderer3.push("<!--[-->");
            $$renderer3.push(`<div class="caption-label svelte-7anmrz">${escape_html(entry.caption)}</div>`);
          } else {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]--></button> `);
          if (interactive) {
            $$renderer3.push("<!--[-->");
            $$renderer3.push(`<button class="delete-button svelte-7anmrz" aria-label="Delete image">`);
            Clear($$renderer3);
            $$renderer3.push(`<!----></button>`);
          } else {
            $$renderer3.push("<!--[!-->");
          }
          $$renderer3.push(`<!--]--></div>`);
        }
        $$renderer3.push(`<!--]--></div></div></div>`);
      }
      $$renderer3.push(`<!--]-->`);
    }
    do {
      $$settled = true;
      $$inner_renderer = $$renderer2.copy();
      $$render_inner($$inner_renderer);
    } while (!$$settled);
    $$renderer2.subsume($$inner_renderer);
    bind_props($$props, { value, selected_index });
  });
}
const Gallery$1 = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: Gallery
}, Symbol.toStringTag, { value: "Module" }));

export { Gallery as G, Gallery$1 as a, handle_save as h };
//# sourceMappingURL=Gallery-CTR9kkPc.js.map
