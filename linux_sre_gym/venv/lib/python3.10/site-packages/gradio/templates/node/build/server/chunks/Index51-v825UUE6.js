import './async-D55cHugf.js';
import { c as spread_props, d as bind_props, f as attr_class, e as ensure_array_like, a as attr } from './index-u8mz_F03.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import { F as File } from './File-2S6P7zIO.js';
import { B as Block } from './Block-JRVGROlG.js';
import { B as BlockLabel } from './BlockLabel-XZM1LqL4.js';
import { I as IconButtonWrapper } from './IconButtonWrapper-DvxA4nj6.js';
import { S as Static } from './index3-BSof0MuO.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';
import './IconButton-D82v5nCM.js';
import './Clear-DH-TDCgr.js';

function ArrowIcon($$renderer) {
  $$renderer.push(`<svg width="100%" height="100%" viewBox="0 0 14 17" version="1.1" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linejoin:round;stroke-miterlimit:2;"><g transform="matrix(1,0,0,1,-10.6667,-7.73588)"><path d="M12.7,24.033C12.256,24.322 11.806,24.339 11.351,24.084C10.896,23.829 10.668,23.434 10.667,22.9L10.667,9.1C10.667,8.567 10.895,8.172 11.351,7.916C11.807,7.66 12.256,7.677 12.7,7.967L23.567,14.867C23.967,15.133 24.167,15.511 24.167,16C24.167,16.489 23.967,16.867 23.567,17.133L12.7,24.033Z" style="fill:currentColor;fill-rule:nonzero;"></path></g></svg>`);
}
function Checkbox($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { value = void 0, disabled, onchange } = $$props;
    $$renderer2.push(`<input${attr("checked", value, true)} type="checkbox"${attr("disabled", disabled, true)}${attr_class("svelte-3g0mcl", void 0, { "disabled": disabled && !value })}/>`);
    bind_props($$props, { value });
  });
}
const FileIcon = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='32'%20height='32'%20viewBox='0%200%2024%2024'%3e%3cpath%20fill='%23888888'%20d='M6%202c-1.1%200-1.99.9-1.99%202L4%2020c0%201.1.89%202%201.99%202H18c1.1%200%202-.9%202-2V8l-6-6H6zm7%207V3.5L18.5%209H13z'/%3e%3c/svg%3e";
const FolderIcon = "data:image/svg+xml,%3c?xml%20version='1.0'%20encoding='UTF-8'%20standalone='no'?%3e%3csvg%20viewBox='0%200%2032%2032'%20version='1.1'%20id='svg7'%20sodipodi:docname='light-folder-new.svg'%20inkscape:version='1.3.2%20(091e20e,%202023-11-25)'%20xmlns:inkscape='http://www.inkscape.org/namespaces/inkscape'%20xmlns:sodipodi='http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd'%20xmlns='http://www.w3.org/2000/svg'%20xmlns:svg='http://www.w3.org/2000/svg'%3e%3csodipodi:namedview%20id='namedview7'%20pagecolor='%23ffffff'%20bordercolor='%23000000'%20borderopacity='0.25'%20inkscape:showpageshadow='2'%20inkscape:pageopacity='0.0'%20inkscape:pagecheckerboard='0'%20inkscape:deskcolor='%23d1d1d1'%20inkscape:zoom='7.375'%20inkscape:cx='15.932203'%20inkscape:cy='16'%20inkscape:window-width='1312'%20inkscape:window-height='529'%20inkscape:window-x='0'%20inkscape:window-y='38'%20inkscape:window-maximized='0'%20inkscape:current-layer='svg7'%20/%3e%3cdefs%20id='defs6'%3e%3cclipPath%20id='clipPath1'%3e%3cpath%20d='m69.63%2012.145h-.052c-22.727-.292-46.47%204.077-46.709%204.122-2.424.451-4.946%202.974-5.397%205.397-.044.237-4.414%2023.983-4.122%2046.71-.292%2022.777%204.078%2046.523%204.122%2046.761.451%202.423%202.974%204.945%205.398%205.398.237.044%2023.982%204.413%2046.709%204.121%2022.779.292%2046.524-4.077%2046.761-4.121%202.423-.452%204.946-2.976%205.398-5.399.044-.236%204.413-23.981%204.121-46.709.292-22.777-4.077-46.523-4.121-46.761-.453-2.423-2.976-4.946-5.398-5.397-.238-.045-23.984-4.414-46.71-4.122'%20id='path1'%20/%3e%3c/clipPath%3e%3clinearGradient%20gradientUnits='userSpaceOnUse'%20y2='352.98'%20x2='-601.15'%20y1='663.95'%20x1='-591.02'%20id='2'%3e%3cstop%20stop-color='%23a0a0a0'%20id='stop1'%20/%3e%3cstop%20offset='1'%20stop-color='%23aaa'%20id='stop2'%20/%3e%3c/linearGradient%3e%3clinearGradient%20gradientUnits='userSpaceOnUse'%20y2='354.29'%20x2='-704.05'%20y1='647.77'%20x1='-701.19'%20id='1'%3e%3cstop%20stop-color='%23acabab'%20id='stop3'%20/%3e%3cstop%20offset='1'%20stop-color='%23d4d4d4'%20id='stop4'%20/%3e%3c/linearGradient%3e%3clinearGradient%20id='0'%20x1='59.12'%20y1='-19.888'%20x2='59.15'%20y2='-37.783'%20gradientUnits='userSpaceOnUse'%20gradientTransform='matrix(4.17478%200%200%204.16765-1069.7%20447.73)'%3e%3cstop%20stop-color='%23a0a0a0'%20id='stop5'%20/%3e%3cstop%20offset='1'%20stop-color='%23bdbdbd'%20id='stop6'%20/%3e%3c/linearGradient%3e%3c/defs%3e%3cg%20transform='matrix(.07089%200%200%20.07017%2023.295-40.67)'%20fill='%2360aae5'%20id='g7'%20style='fill:%23888888;fill-opacity:1'%3e%3cpath%20transform='matrix(.7872%200%200%20.79524%20415.34%20430.11)'%20d='m-884.1%20294.78c-4.626%200-8.349%203.718-8.349%208.335v161.41l468.19%201v-121.2c0-4.618-3.724-8.335-8.35-8.335h-272.65c-8.51.751-9.607-.377-13.812-5.981-5.964-7.968-14.969-21.443-20.84-29.21-4.712-6.805-5.477-6.02-13.292-6.02z'%20fill='url(%230)'%20color='%23000'%20id='path6'%20style='fill:%23888888;fill-opacity:1'%20/%3e%3crect%20transform='matrix(.7872%200%200%20.79524%20415.34%20430.11)'%20y='356.85'%20x='-890.28'%20height='295.13'%20width='463.85'%20fill='url(%231)'%20stroke='url(%231)'%20stroke-width='2.378'%20rx='9.63'%20id='rect6'%20style='fill:%23888888;fill-opacity:1'%20/%3e%3crect%20width='463.85'%20height='295.13'%20x='-890.28'%20y='356.85'%20transform='matrix(.7872%200%200%20.79524%20415.34%20430.11)'%20fill='none'%20stroke='url(%232)'%20stroke-linejoin='round'%20stroke-linecap='round'%20stroke-width='5.376'%20rx='9.63'%20id='rect7'%20style='fill:%23888888;fill-opacity:1'%20/%3e%3c/g%3e%3c/svg%3e";
function FileTree($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      path = [],
      index_path = [],
      selected_files = [],
      selected_folders = [],
      is_selected_entirely = false,
      interactive,
      selectable = false,
      ls_fn,
      file_count = "multiple",
      valid_for_selection,
      oncheck,
      onselect
    } = $$props;
    let content = [];
    let opened_folders = [];
    const open_folder = (i) => {
      if (!opened_folders.includes(i)) {
        opened_folders = [...opened_folders, i];
      }
    };
    (async () => {
      content = await ls_fn(path);
      if (valid_for_selection) {
        content = [{ name: ".", type: "file" }, ...content];
      }
      opened_folders = content.map((x, i) => x.type === "folder" && (is_selected_entirely || selected_files.some((y) => y[0] === x.name)) ? i : null).filter((x) => x !== null);
    })();
    function handle_select(full_index_path, item_path, _type) {
      onselect?.({
        index: full_index_path,
        value: item_path.join("/"),
        selected: true
      });
    }
    $$renderer2.push(`<ul${attr_class("svelte-6im0c4", void 0, { "no-checkboxes": !interactive, "root": path.length === 0 })}><!--[-->`);
    const each_array = ensure_array_like(content);
    for (let i = 0, $$length = each_array.length; i < $$length; i++) {
      let { type, name, valid } = each_array[i];
      const is_selected = (type === "file" ? selected_files : selected_folders).some((x) => x[0] === name && x.length === 1);
      $$renderer2.push(`<li class="svelte-6im0c4"><span${attr_class("wrap svelte-6im0c4", void 0, {
        "selected": !interactive && is_selected,
        "selectable": selectable
      })}${attr("role", selectable ? "button" : void 0)}${attr("tabindex", selectable ? 0 : void 0)}>`);
      if (interactive) {
        $$renderer2.push("<!--[-->");
        if (type === "folder" && file_count === "single") {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<span class="no-checkbox svelte-6im0c4" aria-hidden="true"></span>`);
        } else {
          $$renderer2.push("<!--[!-->");
          Checkbox($$renderer2, {
            disabled: false,
            value: is_selected,
            onchange: (checked) => {
              oncheck?.({ path: [...path, name], checked, type });
              if (selectable) {
                handle_select([...index_path, i], [...path, name]);
              }
              if (type === "folder" && checked) {
                open_folder(i);
              }
            }
          });
        }
        $$renderer2.push(`<!--]-->`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--> `);
      if (type === "folder") {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<span${attr_class("icon svelte-6im0c4", void 0, { "hidden": !opened_folders.includes(i) })} role="button" aria-label="expand directory" tabindex="0">`);
        ArrowIcon($$renderer2);
        $$renderer2.push(`<!----></span>`);
      } else {
        $$renderer2.push("<!--[!-->");
        $$renderer2.push(`<span class="file-icon svelte-6im0c4"><img${attr("src", name === "." ? FolderIcon : FileIcon)} alt="file icon" class="svelte-6im0c4"/></span>`);
      }
      $$renderer2.push(`<!--]--> <span class="item-name svelte-6im0c4">${escape_html(name)}</span></span> `);
      if (type === "folder" && opened_folders.includes(i)) {
        $$renderer2.push("<!--[-->");
        FileTree($$renderer2, {
          path: [...path, name],
          index_path: [...index_path, i],
          selected_files: selected_files.filter((x) => x[0] === name).map((x) => x.slice(1)),
          selected_folders: selected_folders.filter((x) => x[0] === name).map((x) => x.slice(1)),
          is_selected_entirely: selected_folders.some((x) => x[0] === name && x.length === 1),
          interactive,
          selectable,
          ls_fn,
          file_count,
          valid_for_selection: valid ?? false,
          oncheck,
          onselect
        });
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></li>`);
    }
    $$renderer2.push(`<!--]--></ul>`);
  });
}
function DirectoryExplorer($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      interactive,
      file_count = "multiple",
      value = void 0,
      selectable = false,
      ls_fn,
      oninput,
      onselect
    } = $$props;
    let selected_folders = [];
    const paths_equal = (path, path_2) => {
      return path.join("/") === path_2.join("/");
    };
    const path_in_set = (path, set) => {
      return set.some((x) => paths_equal(x, path));
    };
    const path_inside = (path, path_2) => {
      return path.join("/").startsWith(path_2.join("/"));
    };
    $$renderer2.push(`<div class="file-wrap svelte-1gs1iku">`);
    FileTree($$renderer2, {
      path: [],
      selected_files: value,
      selected_folders,
      interactive,
      selectable,
      ls_fn,
      file_count,
      valid_for_selection: false,
      oncheck: (detail) => {
        const { path, checked, type } = detail;
        if (checked) {
          if (file_count === "single") {
            value = [path];
          } else if (type === "folder") {
            if (!path_in_set(path, selected_folders)) {
              selected_folders = [...selected_folders, path];
            }
          } else {
            if (!path_in_set(path, value)) {
              value = [...value, path];
            }
          }
        } else {
          selected_folders = selected_folders.filter((folder) => !path_inside(path, folder));
          if (type === "folder") {
            selected_folders = selected_folders.filter((folder) => !path_inside(folder, path));
            value = value.filter((file) => !path_inside(file, path));
          } else {
            value = value.filter((x) => !paths_equal(x, path));
          }
        }
        oninput?.();
      },
      onselect
    });
    $$renderer2.push(`<!----></div>`);
    bind_props($$props, { value });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    const { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props, { value: [] });
    gradio.props.value;
    [
      gradio.props.root_dir,
      gradio.props.glob,
      gradio.props.ignore_glob
    ];
    let $$settled = true;
    let $$inner_renderer;
    function $$render_inner($$renderer3) {
      Block($$renderer3, {
        visible: gradio.shared.visible,
        variant: gradio.props.value === null ? "dashed" : "solid",
        border_mode: "base",
        padding: false,
        elem_id: gradio.shared.elem_id,
        elem_classes: gradio.shared.elem_classes,
        container: gradio.shared.container,
        scale: gradio.shared.scale,
        min_width: gradio.shared.min_width,
        allow_overflow: true,
        overflow_behavior: "auto",
        height: gradio.props.height,
        max_height: gradio.props.max_height,
        min_height: gradio.props.min_height,
        children: ($$renderer4) => {
          Static($$renderer4, spread_props([
            gradio.shared.loading_status,
            {
              autoscroll: gradio.shared.autoscroll,
              i18n: gradio.i18n,
              on_clear_status: () => gradio.dispatch("clear_status", gradio.shared.loading_status)
            }
          ]));
          $$renderer4.push(`<!----> `);
          if (gradio.shared.show_label && gradio.props.buttons && gradio.props.buttons.length > 0) {
            $$renderer4.push("<!--[-->");
            IconButtonWrapper($$renderer4, {
              buttons: gradio.props.buttons,
              on_custom_button_click: (id) => {
                gradio.dispatch("custom_button_click", { id });
              }
            });
          } else {
            $$renderer4.push("<!--[!-->");
          }
          $$renderer4.push(`<!--]--> `);
          BlockLabel($$renderer4, {
            show_label: gradio.shared.show_label,
            Icon: File,
            label: gradio.shared.label || "FileExplorer",
            float: false
          });
          $$renderer4.push(`<!----> <!---->`);
          {
            DirectoryExplorer($$renderer4, {
              file_count: gradio.props.file_count,
              interactive: gradio.shared.interactive,
              selectable: gradio.props._selectable,
              ls_fn: gradio.shared.server.ls,
              oninput: () => gradio.dispatch("input"),
              onselect: (detail) => gradio.dispatch("select", detail),
              get value() {
                return gradio.props.value;
              },
              set value($$value) {
                gradio.props.value = $$value;
                $$settled = false;
              }
            });
          }
          $$renderer4.push(`<!---->`);
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

export { Index as default };
//# sourceMappingURL=Index51-v825UUE6.js.map
