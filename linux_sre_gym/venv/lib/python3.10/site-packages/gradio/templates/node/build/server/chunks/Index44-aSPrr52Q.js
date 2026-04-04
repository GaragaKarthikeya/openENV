import './async-D55cHugf.js';
import { c as spread_props, k as await_block, d as bind_props, e as ensure_array_like, g as attr_style, f as attr_class, i as stringify, j as clsx } from './index-u8mz_F03.js';
import { B as Block } from './Block-JRVGROlG.js';
import './2-BAeILJ1g.js';
import { G as Gradio } from './utils.svelte-aYomt3KL.js';
import Example from './Example2-8iT2brQo.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';

function MountExample($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { component, runtime, $$slots, $$events, ...rest } = $$props;
    $$renderer2.push(`<span></span>`);
  });
}
function Dataset($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let {
      components,
      component_props,
      load_component,
      headers,
      samples,
      sample_labels = null,
      value = null,
      root,
      proxy_url,
      samples_per_page = 10,
      onclick,
      onselect,
      layout = null
    } = $$props;
    let samples_dir = proxy_url ? `/proxy=${proxy_url}file=` : `${root}/file=`;
    let current_hover = -1;
    let gallery = (components.length < 2 || sample_labels !== null) && layout !== "table";
    let effective_samples = (() => {
      if (sample_labels) {
        return sample_labels.map((e) => [e]);
      }
      return samples ?? [];
    })();
    let page = /* @__PURE__ */ (() => {
      return 0;
    })();
    let paginate = effective_samples.length > samples_per_page;
    let selected_samples = (() => {
      if (paginate) {
        return effective_samples.slice(page * samples_per_page, (page + 1) * samples_per_page);
      }
      return effective_samples.slice();
    })();
    let page_count = Math.ceil(effective_samples.length / samples_per_page);
    let visible_pages = (() => {
      if (!paginate) return [];
      let pages = [];
      [0, page, page_count - 1].forEach((anchor) => {
        for (let i = anchor - 2; i <= anchor + 2; i++) {
          if (i >= 0 && i < page_count && !pages.includes(i)) {
            if (pages.length > 0 && i - pages[pages.length - 1] > 1) {
              pages.push(-1);
            }
            pages.push(i);
          }
        }
      });
      return pages;
    })();
    let component_meta = [];
    async function get_component_meta(selected_samples_json2) {
      const _selected_samples = JSON.parse(selected_samples_json2);
      component_meta = await Promise.all(_selected_samples && _selected_samples.map(async (sample_row) => await Promise.all(sample_row.map(async (sample_cell, j) => {
        const loaded = load_component(components[j].name, "example", components[j].class_id);
        return {
          value: sample_cell,
          component: loaded.component,
          runtime: loaded.runtime
        };
      }))));
    }
    let selected_samples_json = JSON.stringify(selected_samples || []);
    await_block($$renderer2, get_component_meta(selected_samples_json), () => {
    }, (_) => {
      if (gallery) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<div class="gallery svelte-16f20a1"><!--[-->`);
        const each_array = ensure_array_like(selected_samples);
        for (let i = 0, $$length = each_array.length; i < $$length; i++) {
          let sample_row = each_array[i];
          if (sample_row[0] != null) {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`<button class="gallery-item svelte-16f20a1">`);
            if (sample_labels) {
              $$renderer2.push("<!--[-->");
              Example($$renderer2, {
                value: sample_row[0],
                selected: current_hover === i,
                type: "gallery"
              });
            } else {
              $$renderer2.push("<!--[!-->");
              if (component_meta.length) {
                $$renderer2.push("<!--[-->");
                await_block($$renderer2, Promise.all([component_meta[0][0].component, component_meta[0][0].runtime]), () => {
                }, ([component, runtime]) => {
                  $$renderer2.push(`<!---->`);
                  {
                    MountExample($$renderer2, spread_props([
                      { component, runtime },
                      component_props[0],
                      {
                        value: sample_row[0],
                        samples_dir,
                        type: "gallery",
                        selected: current_hover === i,
                        index: i,
                        root
                      }
                    ]));
                  }
                  $$renderer2.push(`<!---->`);
                });
                $$renderer2.push(`<!--]-->`);
              } else {
                $$renderer2.push("<!--[!-->");
              }
              $$renderer2.push(`<!--]-->`);
            }
            $$renderer2.push(`<!--]--></button>`);
          } else {
            $$renderer2.push("<!--[!-->");
          }
          $$renderer2.push(`<!--]-->`);
        }
        $$renderer2.push(`<!--]--></div>`);
      } else {
        $$renderer2.push("<!--[!-->");
        if (selected_samples.length > 0) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<div class="table-wrap svelte-16f20a1"><table tabindex="0" role="grid" class="svelte-16f20a1"><thead><tr class="tr-head svelte-16f20a1"><!--[-->`);
          const each_array_1 = ensure_array_like(headers);
          for (let $$index_1 = 0, $$length = each_array_1.length; $$index_1 < $$length; $$index_1++) {
            let header = each_array_1[$$index_1];
            $$renderer2.push(`<th class="svelte-16f20a1">${escape_html(header)}</th>`);
          }
          $$renderer2.push(`<!--]--></tr></thead><tbody><!--[-->`);
          const each_array_2 = ensure_array_like(component_meta);
          for (let i = 0, $$length = each_array_2.length; i < $$length; i++) {
            let sample_row = each_array_2[i];
            $$renderer2.push(`<tr class="tr-body svelte-16f20a1"><!--[-->`);
            const each_array_3 = ensure_array_like(sample_row);
            for (let j = 0, $$length2 = each_array_3.length; j < $$length2; j++) {
              let { value: value2, component, runtime } = each_array_3[j];
              const component_name = components[j];
              if (component_name !== void 0) {
                $$renderer2.push("<!--[-->");
                $$renderer2.push(`<td${attr_style(`max-width: ${stringify(component_name === "textbox" ? "35ch" : "auto")}`)}${attr_class(clsx(component_name), "svelte-16f20a1")}>`);
                await_block($$renderer2, Promise.all([component, runtime]), () => {
                }, ([component2, runtime2]) => {
                  MountExample($$renderer2, spread_props([
                    { component: component2, runtime: runtime2 },
                    component_props[j],
                    {
                      value: value2,
                      samples_dir,
                      type: "table",
                      selected: current_hover === i,
                      index: i,
                      root
                    }
                  ]));
                });
                $$renderer2.push(`<!--]--></td>`);
              } else {
                $$renderer2.push("<!--[!-->");
              }
              $$renderer2.push(`<!--]-->`);
            }
            $$renderer2.push(`<!--]--></tr>`);
          }
          $$renderer2.push(`<!--]--></tbody></table></div>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]--> `);
      if (paginate) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<div class="paginate svelte-16f20a1">Pages: <!--[-->`);
        const each_array_4 = ensure_array_like(visible_pages);
        for (let $$index_4 = 0, $$length = each_array_4.length; $$index_4 < $$length; $$index_4++) {
          let visible_page = each_array_4[$$index_4];
          if (visible_page === -1) {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`<div>...</div>`);
          } else {
            $$renderer2.push("<!--[!-->");
            $$renderer2.push(`<button${attr_class("svelte-16f20a1", void 0, { "current-page": page === visible_page })}>${escape_html(visible_page + 1)}</button>`);
          }
          $$renderer2.push(`<!--]-->`);
        }
        $$renderer2.push(`<!--]--></div>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    });
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, { value });
  });
}
function Index($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let { $$slots, $$events, ...props } = $$props;
    const gradio = new Gradio(props);
    let samples = gradio.props.samples ?? [];
    Block($$renderer2, {
      visible: gradio.shared.visible,
      padding: false,
      elem_id: gradio.shared.elem_id,
      elem_classes: gradio.shared.elem_classes,
      scale: gradio.shared.scale,
      min_width: gradio.shared.min_width,
      allow_overflow: false,
      container: false,
      children: ($$renderer3) => {
        if (gradio.shared.show_label) {
          $$renderer3.push("<!--[-->");
          $$renderer3.push(`<div class="label svelte-bnxc4d"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32" class="svelte-bnxc4d"><path fill="currentColor" d="M10 6h18v2H10zm0 18h18v2H10zm0-9h18v2H10zm-6 0h2v2H4zm0-9h2v2H4zm0 18h2v2H4z"></path></svg> ${escape_html(gradio.shared.label || "Examples")}</div>`);
        } else {
          $$renderer3.push("<!--[!-->");
        }
        $$renderer3.push(`<!--]--> `);
        Dataset($$renderer3, spread_props([
          {
            onclick: (d) => (gradio.props.value = d.index, gradio.dispatch("click", gradio.props.value)),
            onselect: (data) => gradio.dispatch("select", data),
            load_component: gradio.shared.load_component,
            samples
          },
          gradio.props
        ]));
        $$renderer3.push(`<!---->`);
      },
      $$slots: { default: true }
    });
  });
}

export { Index as default };
//# sourceMappingURL=Index44-aSPrr52Q.js.map
