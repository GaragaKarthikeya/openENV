import { f as fallback } from './async-D55cHugf.js';
import { l as element, g as attr_style, d as bind_props, s as slot, a as attr, f as attr_class, i as stringify } from './index-u8mz_F03.js';

/* empty css                                         */
function Block($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    let height = fallback($$props["height"], void 0);
    let min_height = fallback($$props["min_height"], void 0);
    let max_height = fallback($$props["max_height"], void 0);
    let width = fallback($$props["width"], void 0);
    let elem_id = fallback($$props["elem_id"], "");
    let elem_classes = fallback($$props["elem_classes"], () => [], true);
    let variant = fallback($$props["variant"], "solid");
    let border_mode = fallback($$props["border_mode"], "base");
    let padding = fallback($$props["padding"], true);
    let type = fallback($$props["type"], "normal");
    let test_id = fallback($$props["test_id"], void 0);
    let explicit_call = fallback($$props["explicit_call"], false);
    let container = fallback($$props["container"], true);
    let visible = fallback($$props["visible"], true);
    let allow_overflow = fallback($$props["allow_overflow"], true);
    let overflow_behavior = fallback($$props["overflow_behavior"], "auto");
    let scale = fallback($$props["scale"], null);
    let min_width = fallback($$props["min_width"], 0);
    let flex = fallback($$props["flex"], false);
    let resizable = fallback($$props["resizable"], false);
    let rtl = fallback($$props["rtl"], false);
    let fullscreen = fallback($$props["fullscreen"], false);
    let label = fallback($$props["label"], void 0);
    let old_fullscreen = fullscreen;
    let element$1;
    let tag = type === "fieldset" ? "fieldset" : "div";
    let placeholder_height = 0;
    let placeholder_width = 0;
    let preexpansionBoundingRect = null;
    function handleKeydown(event) {
      if (fullscreen && event.key === "Escape") {
        fullscreen = false;
      }
    }
    const get_dimension = (dimension_value) => {
      if (dimension_value === void 0) {
        return void 0;
      }
      if (typeof dimension_value === "number") {
        return dimension_value + "px";
      } else if (typeof dimension_value === "string") {
        return dimension_value;
      }
    };
    if (fullscreen !== old_fullscreen) {
      old_fullscreen = fullscreen;
      if (fullscreen) {
        preexpansionBoundingRect = element$1.getBoundingClientRect();
        placeholder_height = element$1.offsetHeight;
        placeholder_width = element$1.offsetWidth;
        window.addEventListener("keydown", handleKeydown);
      } else {
        preexpansionBoundingRect = null;
        window.removeEventListener("keydown", handleKeydown);
      }
    }
    if (!visible) {
      flex = false;
    }
    if (
      // When visible is False, we need to remove the component from the page
      // We can remove it by either modifying the AppTree in Blocks or by hiding the component here
      // We do it here because if visible is updated via an event, only the local state will be updated
      // and we would have to flow the state back up to modify the AppTree
      visible === true || visible === "hidden"
    ) {
      $$renderer2.push("<!--[-->");
      element(
        $$renderer2,
        tag,
        () => {
          $$renderer2.push(`${attr("data-testid", test_id)}${attr("id", elem_id)}${attr_class(`block ${stringify(elem_classes?.join(" ") || "")}`, "svelte-1plpy97", {
            "hidden": visible === "hidden",
            "padded": padding,
            "flex": flex,
            "border_focus": border_mode === "focus",
            "border_contrast": border_mode === "contrast",
            "hide-container": !explicit_call && !container,
            "fullscreen": fullscreen,
            "animating": fullscreen && preexpansionBoundingRect !== null,
            "auto-margin": scale === null
          })}${attr("dir", rtl ? "rtl" : "ltr")}${attr("aria-label", label)}${attr_style("", {
            height: fullscreen ? void 0 : get_dimension(height),
            "min-height": fullscreen ? void 0 : get_dimension(min_height),
            "max-height": fullscreen ? void 0 : get_dimension(max_height),
            "--start-top": preexpansionBoundingRect ? `${preexpansionBoundingRect.top}px` : "0px",
            "--start-left": preexpansionBoundingRect ? `${preexpansionBoundingRect.left}px` : "0px",
            "--start-width": preexpansionBoundingRect ? `${preexpansionBoundingRect.width}px` : "0px",
            "--start-height": preexpansionBoundingRect ? `${preexpansionBoundingRect.height}px` : "0px",
            width: fullscreen ? void 0 : typeof width === "number" ? `calc(min(${width}px, 100%))` : get_dimension(width),
            "border-style": variant,
            overflow: allow_overflow ? overflow_behavior : "hidden",
            "flex-grow": scale,
            "min-width": `calc(min(${min_width}px, 100%))`
          })}`);
        },
        () => {
          $$renderer2.push(`<!--[-->`);
          slot($$renderer2, $$props, "default", {});
          $$renderer2.push(`<!--]--> `);
          if (resizable) {
            $$renderer2.push("<!--[-->");
            $$renderer2.push(`<svg class="resize-handle svelte-1plpy97" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" class="svelte-1plpy97"></line><line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" class="svelte-1plpy97"></line></svg>`);
          } else {
            $$renderer2.push("<!--[!-->");
          }
          $$renderer2.push(`<!--]-->`);
        }
      );
      $$renderer2.push(` `);
      if (fullscreen) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<div class="placeholder svelte-1plpy97"${attr_style("", {
          height: placeholder_height + "px",
          width: placeholder_width + "px"
        })}></div>`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]-->`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]-->`);
    bind_props($$props, {
      height,
      min_height,
      max_height,
      width,
      elem_id,
      elem_classes,
      variant,
      border_mode,
      padding,
      type,
      test_id,
      explicit_call,
      container,
      visible,
      allow_overflow,
      overflow_behavior,
      scale,
      min_width,
      flex,
      resizable,
      rtl,
      fullscreen,
      label
    });
  });
}

export { Block as B };
//# sourceMappingURL=Block-JRVGROlG.js.map
