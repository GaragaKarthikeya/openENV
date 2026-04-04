import { f as fallback } from './async-D55cHugf.js';
import { b as store_get, f as attr_class, g as attr_style, e as ensure_array_like, a as attr, s as slot, i as stringify, u as unsubscribe_stores, d as bind_props } from './index-u8mz_F03.js';
import { t as tick } from './index-server-CQz6EZl_.js';
import { n as navbar_config, $ as $format, d as get_inputs_outputs, e as get_component, f as determine_interactivity } from './2-BAeILJ1g.js';
export { h as changeLocale, i as get_initial_locale, j as is_translation_metadata, l as language_choices, k as load_translations, m as mount_css, p as prefix_css, o as process_langs, s as setupi18n } from './2-BAeILJ1g.js';
import { a as allowed_shared_props } from './utils.svelte-aYomt3KL.js';
export { I as i18n_marker } from './utils.svelte-aYomt3KL.js';
import { e as escape_html } from './escaping-CBnpiEl5.js';
import './context-CBkBucIx.js';
import './index5-BZVOFaHm.js';
import './dev-fallback-B-RpELjM.js';
import './index-Cg-Pg6j3.js';
import './clone-Yk88IHKV.js';

const space_logo = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='10'%20height='10'%20fill='none'%3e%3cpath%20fill='%23FF3270'%20d='M1.93%206.03v2.04h2.04V6.03H1.93Z'/%3e%3cpath%20fill='%23861FFF'%20d='M6.03%206.03v2.04h2.04V6.03H6.03Z'/%3e%3cpath%20fill='%23097EFF'%20d='M1.93%201.93v2.04h2.04V1.93H1.93Z'/%3e%3cpath%20fill='%23000'%20fill-rule='evenodd'%20d='M.5%201.4c0-.5.4-.9.9-.9h3.1a.9.9%200%200%201%20.87.67A2.44%202.44%200%200%201%209.5%202.95c0%20.65-.25%201.24-.67%201.68.39.1.67.46.67.88v3.08c0%20.5-.4.91-.9.91H1.4a.9.9%200%200%201-.9-.9V1.4Zm1.43.53v2.04h2.04V1.93H1.93Zm0%206.14V6.03h2.04v2.04H1.93Zm4.1%200V6.03h2.04v2.04H6.03Zm0-5.12a1.02%201.02%200%201%201%202.04%200%201.02%201.02%200%200%201-2.04%200Z'%20clip-rule='evenodd'/%3e%3cpath%20fill='%23FFD702'%20d='M7.05%201.93a1.02%201.02%200%201%200%200%202.04%201.02%201.02%200%200%200%200-2.04Z'/%3e%3c/svg%3e";
function Embed($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    var $$store_subs;
    let show_navbar, effective_pages;
    let wrapper = $$props["wrapper"];
    let version = $$props["version"];
    let initial_height = $$props["initial_height"];
    let fill_width = $$props["fill_width"];
    let is_embed = $$props["is_embed"];
    let space = $$props["space"];
    let display = $$props["display"];
    let info = $$props["info"];
    let loaded = $$props["loaded"];
    let pages = fallback($$props["pages"], () => [], true);
    let current_page = fallback($$props["current_page"], "");
    let root = $$props["root"];
    let components = fallback($$props["components"], () => [], true);
    let navbar_component = components.find((c) => c.type === "navbar");
    let navbar = navbar_component ? {
      visible: navbar_component.props.visible,
      main_page_name: navbar_component.props.main_page_name,
      value: navbar_component.props.value
    } : null;
    if (navbar) {
      navbar_config.set(navbar);
    }
    function normalize_path(path) {
      let normalized = path.split("?")[0].split("#")[0];
      normalized = normalized.replace(/^\/+|\/+$/g, "");
      return normalized;
    }
    function is_active_route(route, current) {
      if (route.startsWith("http://") || route.startsWith("https://")) {
        return false;
      }
      return normalize_path(route) === normalize_path(current);
    }
    if (store_get($$store_subs ??= {}, "$navbar_config", navbar_config)) {
      navbar = {
        visible: store_get($$store_subs ??= {}, "$navbar_config", navbar_config).visible ?? true,
        main_page_name: store_get($$store_subs ??= {}, "$navbar_config", navbar_config).main_page_name ?? "Home",
        value: store_get($$store_subs ??= {}, "$navbar_config", navbar_config).value ?? null
      };
    }
    show_navbar = pages.length > 1 && (navbar === null || navbar.visible !== false);
    effective_pages = (() => {
      let visible_pages = pages.filter(([route, label, show], index) => {
        if (index === 0 && route === "") {
          return navbar?.main_page_name !== false;
        }
        return show !== false;
      });
      let base_pages = navbar && navbar.main_page_name !== false && navbar.main_page_name !== "Home" ? visible_pages.map(([route, label, show], index) => index === 0 && route === "" && label === "Home" ? [route, navbar.main_page_name] : [route, label]) : visible_pages.map(([route, label]) => [route, label]);
      if (navbar?.value && navbar.value.length > 0) {
        const existing_routes = new Set(base_pages.map(([route]) => route));
        const additional_pages = navbar.value.map(([page_name, page_path]) => [page_path, page_name]).filter(([route]) => !existing_routes.has(route));
        return [...base_pages, ...additional_pages];
      }
      return base_pages;
    })();
    $$renderer2.push(`<div${attr_class(`gradio-container gradio-container-${stringify(version)}`, "svelte-99kmwu", {
      "fill_width": fill_width,
      "embed-container": display,
      "with-info": info
    })} data-iframe-height=""${attr_style("", {
      "min-height": loaded ? "initial" : initial_height,
      "flex-grow": !display ? "1" : "auto"
    })}>`);
    if (show_navbar) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="nav-holder svelte-99kmwu"><nav${attr_class("fillable svelte-99kmwu", void 0, { "fill_width": fill_width })}><!--[-->`);
      const each_array = ensure_array_like(effective_pages);
      for (let i = 0, $$length = each_array.length; i < $$length; i++) {
        let [route, label] = each_array[i];
        $$renderer2.push(`<a${attr("href", route.startsWith("http://") || route.startsWith("https://") ? route : `${root}/${route}`)} data-sveltekit-reload=""${attr("target", route.startsWith("http://") || route.startsWith("https://") ? "_blank" : "_self")}${attr("rel", route.startsWith("http://") || route.startsWith("https://") ? "noopener noreferrer" : "")}${attr_class("svelte-99kmwu", void 0, { "active": is_active_route(route, current_page) })}>${escape_html(label)}</a>`);
      }
      $$renderer2.push(`<!--]--></nav></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--> <div${attr_class("main fillable svelte-99kmwu", void 0, { "fill_width": fill_width, "app": !display && !is_embed })}><!--[-->`);
    slot($$renderer2, $$props, "default", {});
    $$renderer2.push(`<!--]--> <div>`);
    if (display && space && info) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="info svelte-99kmwu"><span class="svelte-99kmwu"><a${attr("href", `https://huggingface.co/spaces/${stringify(space)}`)} class="title svelte-99kmwu">${escape_html(space)}</a></span> <span class="svelte-99kmwu">${escape_html(store_get($$store_subs ??= {}, "$_", $format)("common.built_with"))} <a class="gradio svelte-99kmwu" href="https://gradio.app">Gradio</a>.</span> <span class="svelte-99kmwu">${escape_html(store_get($$store_subs ??= {}, "$_", $format)("common.hosted_on"))} <a class="hf svelte-99kmwu" href="https://huggingface.co/spaces"><span class="space-logo svelte-99kmwu"><img${attr("src", space_logo)} alt="Hugging Face Space" class="svelte-99kmwu"/></span> Spaces</a></span></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div></div></div>`);
    if ($$store_subs) unsubscribe_stores($$store_subs);
    bind_props($$props, {
      wrapper,
      version,
      initial_height,
      fill_width,
      is_embed,
      space,
      display,
      info,
      loaded,
      pages,
      current_page,
      root,
      components
    });
  });
}
var has = Object.prototype.hasOwnProperty;
function find(iter, tar, key) {
  for (key of iter.keys()) {
    if (dequal(key, tar)) return key;
  }
}
function dequal(foo, bar) {
  var ctor, len, tmp;
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
    if (ctor === Set) {
      if (foo.size !== bar.size) {
        return false;
      }
      for (len of foo) {
        tmp = len;
        if (tmp && typeof tmp === "object") {
          tmp = find(bar, tmp);
          if (!tmp) return false;
        }
        if (!bar.has(tmp)) return false;
      }
      return true;
    }
    if (ctor === Map) {
      if (foo.size !== bar.size) {
        return false;
      }
      for (len of foo) {
        tmp = len[0];
        if (tmp && typeof tmp === "object") {
          tmp = find(bar, tmp);
          if (!tmp) return false;
        }
        if (!dequal(len[1], bar.get(tmp))) {
          return false;
        }
      }
      return true;
    }
    if (ctor === ArrayBuffer) {
      foo = new Uint8Array(foo);
      bar = new Uint8Array(bar);
    } else if (ctor === DataView) {
      if ((len = foo.byteLength) === bar.byteLength) {
        while (len-- && foo.getInt8(len) === bar.getInt8(len)) ;
      }
      return len === -1;
    }
    if (ArrayBuffer.isView(foo)) {
      if ((len = foo.byteLength) === bar.byteLength) {
        while (len-- && foo[len] === bar[len]) ;
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
const type_map = { walkthrough: "tabs", walkthroughstep: "tabitem" };
function get_api_url(config) {
  const rootUrl = new URL(config.root);
  const rootPath = rootUrl.pathname.endsWith("/") ? rootUrl.pathname : rootUrl.pathname + "/";
  const apiPrefix = config.api_prefix.startsWith("/") ? config.api_prefix : "/" + config.api_prefix;
  return new URL(rootPath.slice(0, -1) + apiPrefix, rootUrl.origin).toString();
}
class AppTree {
  /** the raw component structure received from the backend */
  #component_payload;
  /** the raw layout node structure received from the backend */
  #layout_payload;
  /** the raw dependency structure received from the backend */
  #dependency_payload;
  /** Need this to set i18n in re-render */
  reactive_formatter = (str) => str;
  /** the config for the app */
  #config;
  client;
  /** the root node of the processed layout tree */
  root;
  root_untracked;
  /** a set of all component IDs that are inputs to dependencies */
  #input_ids = /* @__PURE__ */ new Set();
  /** a set of all component IDs that are outputs of dependencies */
  #output_ids = /* @__PURE__ */ new Set();
  /** A list of components that are currently loading */
  #pending_components = [];
  #get_callbacks = /* @__PURE__ */ new Map();
  #set_callbacks = /* @__PURE__ */ new Map();
  #pending_updates = /* @__PURE__ */ new Map();
  #event_dispatcher;
  component_ids;
  initial_tabs = {};
  components_to_register = /* @__PURE__ */ new Set();
  ready;
  ready_resolve;
  resolved = false;
  #hidden_on_startup = /* @__PURE__ */ new Set();
  constructor(components, layout, dependencies, config, app, reactive_formatter, event_dispatcher) {
    this.ready = new Promise((resolve) => {
      this.ready_resolve = resolve;
    });
    this.reactive_formatter = reactive_formatter;
    const api_url = get_api_url(config);
    this.#config = { ...config, api_url };
    this.#component_payload = components;
    this.#layout_payload = layout;
    this.#dependency_payload = dependencies;
    this.#event_dispatcher = event_dispatcher;
    this.root = this.create_node({ id: layout.id, children: [] }, /* @__PURE__ */ new Map(), true);
    for (const comp of components) {
      if (comp.props.visible != false) this.components_to_register.add(comp.id);
    }
    this.client = app;
    this.prepare();
    const component_map = components.reduce(
      (map, comp) => {
        map.set(comp.id, comp);
        return map;
      },
      /* @__PURE__ */ new Map()
    );
    this.root.children = this.#layout_payload.children.map((node) => this.traverse(node, (node2) => {
      const new_node = this.create_node(node2, component_map, false, this.reactive_formatter);
      return new_node;
    }));
    this.component_ids = components.map((c) => c.id);
    this.initial_tabs = {};
    gather_initial_tabs(this.root, this.initial_tabs);
    this.postprocess(this.root);
    this.root_untracked = this.root;
  }
  reload(components, layout, dependencies, config) {
    this.#layout_payload = layout;
    this.#component_payload = components;
    const api_url = get_api_url(config);
    this.#config = { ...config, api_url };
    this.#dependency_payload = dependencies;
    this.root = this.create_node({ id: layout.id, children: [] }, /* @__PURE__ */ new Map(), true);
    for (const comp of components) {
      if (comp.props.visible != false) this.components_to_register.add(comp.id);
    }
    this.prepare();
    const component_map = components.reduce(
      (map, comp) => {
        map.set(comp.id, comp);
        return map;
      },
      /* @__PURE__ */ new Map()
    );
    this.root.children = this.#layout_payload.children.map((node) => this.traverse(node, (node2) => {
      const new_node = this.create_node(node2, component_map, false, this.reactive_formatter);
      return new_node;
    }));
    this.component_ids = components.map((c) => c.id);
    this.initial_tabs = {};
    gather_initial_tabs(this.root, this.initial_tabs);
    this.postprocess(this.root);
  }
  /**
   * Registers a component with its ID and data callbacks
   * @param id the ID of the component
   * @param _set_data the set data callback
   * @param _get_data the get data callback
   */
  register_component(id, _set_data, _get_data) {
    this.#set_callbacks.set(id, _set_data);
    this.#get_callbacks.set(id, _get_data);
    this.components_to_register.delete(id);
    const pending = this.#pending_updates.get(id);
    if (pending) {
      this.#pending_updates.delete(id);
      tick().then(() => {
        const _set = this.#set_callbacks.get(id);
        if (_set) _set(pending);
      });
    }
    if (this.components_to_register.size === 0 && !this.resolved) {
      this.resolved = true;
      this.ready_resolve();
    }
  }
  /**
   * Preprocess the payloads to get the correct state read to build the tree
   */
  prepare() {
    const [inputs, outputs] = get_inputs_outputs(this.#dependency_payload);
    this.#input_ids = inputs;
    this.#output_ids = outputs;
  }
  /** Processes the layout payload into a tree of components */
  process() {
  }
  postprocess(tree) {
    this.root = this.traverse(tree, [
      (node) => handle_visibility(node, this.#config.api_url),
      (node) => untrack_children_of_invisible_parents(node, this.components_to_register),
      (node) => apply_initial_tabs(node, this.initial_tabs),
      (node) => this.find_attached_events(node, this.#dependency_payload),
      (node) => untrack_children_of_closed_accordions_or_inactive_tabs(node, this.components_to_register, this.#hidden_on_startup)
    ]);
  }
  find_attached_events(node, dependencies) {
    const attached_events = dependencies.filter((dep) => dep.targets.find(([id]) => id === node.id)).map((dep) => {
      const target = dep.targets.find(([id]) => id === node.id);
      return target ? target[1] : null;
    }).filter(Boolean);
    node.props.shared_props.attached_events = attached_events;
    return node;
  }
  /**
   * Traverses the layout tree and applies a callback to each node
   * @param node the current layout node
   * @param visit the callback to apply to each node
   * @returns the return value of the callback, with a `children` property added for any child nodes
   */
  traverse(node, visit) {
    function single_visit(node2, visit2, traverse_fn) {
      const result = visit2(node2);
      if ("children" in node2 && node2.children.length > 0) {
        result.children = node2.children?.map((child) => traverse_fn(child, visit2)) || [];
      }
      return result;
    }
    if (Array.isArray(visit)) {
      let result = node;
      for (const v of visit) {
        result = single_visit(result, v, this.traverse.bind(this));
      }
      return result;
    } else {
      return single_visit(node, visit, this.traverse.bind(this));
    }
  }
  /**
   * Creates a processed component node from a layout node
   * @param opts the layout node options
   * @param root whether this is the root node
   * @returns the processed component node
   */
  create_node(opts, component_map, root = false, reactive_formatter) {
    let component;
    if (!root) {
      component = component_map.get(opts.id);
    } else {
      component = {
        type: "column",
        id: opts.id,
        // @ts-ignore
        props: {
          visible: true,
          root: "",
          theme_mode: "light",
          scale: this.#config.fill_height ? 1 : null
        },
        component_class_id: "column",
        key: null
      };
    }
    if (!component) {
      throw new Error(`Component with ID ${opts.id} not found`);
    }
    if (reactive_formatter) {
      component.props.i18n = reactive_formatter;
    }
    const processed_props = gather_props(opts.id, component.props, [this.#input_ids, this.#output_ids], this.client, this.#config.api_url, {
      ...this.#config,
      register_component: this.register_component.bind(this),
      dispatcher: this.#event_dispatcher.bind(this)
    });
    const type = type_map[component.type] || component.type;
    const loading_component = processed_props.shared_props.visible !== false ? get_component(component.type, component.component_class_id, this.#config.api_url || "") : null;
    const node = {
      id: opts.id,
      type,
      props: processed_props,
      children: [],
      show_progress_on: null,
      component_class_id: component.component_class_id || component.type,
      component: processed_props.shared_props.visible !== false ? loading_component?.component || null : null,
      runtime: loading_component?.runtime || false,
      key: component.key,
      rendered_in: component.rendered_in,
      documentation: component.documentation,
      original_visibility: processed_props.shared_props.visible
    };
    return node;
  }
  rerender(components, layout) {
    const component_map = components.reduce(
      (map, comp) => {
        map.set(comp.id, comp);
        return map;
      },
      /* @__PURE__ */ new Map()
    );
    const _subtree = this.traverse(layout, (node) => {
      const new_node = this.create_node(node, component_map, false, this.reactive_formatter);
      return new_node;
    });
    gather_initial_tabs(_subtree, this.initial_tabs);
    const subtree = this.traverse(_subtree, (node) => apply_initial_tabs(node, this.initial_tabs));
    const n = find_node_by_id(this.root, subtree.id);
    if (!n) {
      throw new Error("Rerender failed: root node not found in current tree");
    }
    n.children = subtree.children;
  }
  /*
   * Updates the state of a component by its ID
   * @param id the ID of the component to update
   * @param new_state the new state to set
   * */
  async update_state(id, new_state, check_visibility = true) {
    const node = find_node_by_id(this.root, id);
    let already_updated_visibility = false;
    if (check_visibility && !node?.component) {
      await tick();
      this.root = this.traverse(this.root, [
        //@ts-ignore
        (n) => set_visibility_for_updated_node(n, id, new_state.visible),
        //@ts-ignore
        (n) => handle_visibility(n, this.#config.api_url)
      ]);
      await tick();
      already_updated_visibility = true;
    }
    const _set_data = this.#set_callbacks.get(id);
    if (!_set_data) {
      const old_value = node?.props.props.value;
      const new_props = create_props_shared_props(new_state);
      for (const key in new_props.shared_props) {
        node.props.shared_props[key] = new_props.shared_props[key];
      }
      for (const key in new_props.props) {
        node.props.props[key] = new_props.props[key];
      }
      const existing = this.#pending_updates.get(id) || {};
      this.#pending_updates.set(id, { ...existing, ...new_state });
      if ("value" in new_state && !dequal(old_value, new_state.value)) {
        this.#event_dispatcher(id, "change", null);
      }
      if (node?.type === "tabitem") {
        this.#update_parent_tabs_initial_tab(id, node);
      }
    } else if (_set_data) {
      _set_data(new_state);
    }
    if (!check_visibility || already_updated_visibility) return;
    await tick();
    await this.update_visibility(node, new_state);
  }
  async update_visibility(node, new_state) {
    node.children.forEach((child) => {
      const _set_data = this.#set_callbacks.get(child.id);
      if (_set_data) {
        _set_data(new_state);
      }
      this.update_visibility(child, new_state);
    });
  }
  /**
   * Updates the parent Tabs component's initial_tabs when a non-mounted
   * tabitem's props change. This ensures the tab button (rendered by
   * the Tabs component) reflects the updated state even though the
   * TabItem component itself is not mounted.
   */
  #update_parent_tabs_initial_tab(id, node) {
    const parent = find_parent(this.root, id);
    if (!parent || parent.type !== "tabs") return;
    const initial_tabs = parent.props.props.initial_tabs;
    if (!initial_tabs) return;
    const tab_index = initial_tabs.findIndex((t) => t.component_id === node.id);
    if (tab_index === -1) return;
    const i18n = node.props.props.i18n;
    const raw_label = node.props.shared_props.label;
    const visible = "original_visibility" in node ? node.original_visibility : node.props.shared_props.visible;
    initial_tabs[tab_index] = {
      label: i18n ? i18n(raw_label) : raw_label,
      id: node.props.props.id,
      elem_id: node.props.shared_props.elem_id,
      visible,
      interactive: node.props.shared_props.interactive,
      scale: node.props.shared_props.scale || null,
      component_id: node.id
    };
    parent.props.props.initial_tabs = [...initial_tabs];
    const parent_set_data = this.#set_callbacks.get(parent.id);
    if (parent_set_data) {
      parent_set_data({ initial_tabs: parent.props.props.initial_tabs });
    }
  }
  /**
   * Gets the current state of a component by its ID
   * @param id the ID of the component to get the state of
   * @returns the current state of the component, or null if not found
   */
  async get_state(id) {
    const _get_data = this.#get_callbacks.get(id);
    const component = find_node_by_id(this.root, id);
    if (!_get_data && !component) return null;
    if (_get_data) return await _get_data();
    if (component) return Promise.resolve({ value: component.props.props.value });
    return null;
  }
  async render_previously_invisible_children(id) {
    const node = find_node_by_id(this.root, id);
    if (!node) return;
    if (!this.#hidden_on_startup.has(node.id) && !has_hidden_descendants(node, this.#hidden_on_startup)) {
      return;
    }
    make_visible_if_not_rendered(node, this.#hidden_on_startup, true);
    load_components(node, this.#config.api_url);
  }
}
function make_visible_if_not_rendered(node, hidden_on_startup, is_target_node = false) {
  node.props.shared_props.visible = hidden_on_startup.has(node.id) ? true : node.props.shared_props.visible;
  if (node.type === "tabs") {
    const selectedId = node.props.props.selected ?? node.props.props.initial_tabs?.[0]?.id;
    node.children.forEach((child) => {
      if (child.type === "tabitem" && (child.props.props.id === selectedId || child.id === selectedId)) {
        make_visible_if_not_rendered(child, hidden_on_startup, false);
      }
    });
  } else if (node.type === "accordion" && node.props.props.open === false && !is_target_node) ;
  else {
    node.children.forEach((child) => {
      make_visible_if_not_rendered(child, hidden_on_startup, false);
    });
  }
}
function has_hidden_descendants(node, hidden_on_startup) {
  for (const child of node.children) {
    if (hidden_on_startup.has(child.id)) return true;
    if (has_hidden_descendants(child, hidden_on_startup)) return true;
  }
  return false;
}
function load_components(node, api_url) {
  if (node.props.shared_props.visible && !node.component) {
    node.component = get_component(node.type, node.component_class_id, api_url);
  }
  node.children.forEach((child) => load_components(child, api_url));
}
function process_server_fn(id, server_fns, app) {
  if (!server_fns) {
    return {};
  }
  return server_fns.reduce(
    (acc, fn) => {
      acc[fn] = async (...args) => {
        if (args.length === 1) {
          args = args[0];
        }
        const result = await app.component_server(id, fn, args);
        return result;
      };
      return acc;
    },
    {}
  );
}
function create_props_shared_props(props) {
  const _shared_props = {};
  const _props = {};
  for (const key in props) {
    if (key === "id" || key === "autoscroll") {
      _props[key] = props[key];
    } else if (allowed_shared_props.includes(key)) {
      const _key = key;
      _shared_props[_key] = props[key];
    } else {
      _props[key] = props[key];
    }
  }
  return { shared_props: _shared_props, props: _props };
}
function gather_props(id, props, dependencies, client, api_url, additional = {}) {
  const { shared_props: _shared_props, props: _props } = create_props_shared_props(props);
  _shared_props.server = process_server_fn(id, props.server_fns, client);
  for (const key in additional) {
    if (allowed_shared_props.includes(key)) {
      const _key = key;
      _shared_props[_key] = additional[key];
    } else {
      _props[key] = additional[key];
    }
  }
  _shared_props.client = client;
  _shared_props.id = id;
  _shared_props.interactive = determine_interactivity(id, _shared_props.interactive, _props.value, dependencies);
  _shared_props.load_component = (name, variant, component_class_id) => get_component(name, component_class_id || "", api_url, variant);
  _shared_props.visible = _shared_props.visible === void 0 ? true : _shared_props.visible;
  _shared_props.loading_status = {};
  return { shared_props: _shared_props, props: _props };
}
function handle_visibility(node, api_url) {
  if (node.props.shared_props.visible && !node.component) {
    const loading_component = get_component(node.type, node.component_class_id, api_url);
    const result = {
      ...node,
      component: loading_component.component,
      children: []
    };
    if (node.children) {
      result.children = node.children.map((child) => handle_visibility(child, api_url));
    }
    return result;
  } else {
    return node;
  }
}
function set_visibility_for_updated_node(node, id, visible) {
  if (node.id == id) {
    node.props.shared_props.visible = visible;
  }
  return node;
}
function _untrack(node, components_to_register) {
  components_to_register.delete(node.id);
  if (node.children) {
    node.children.forEach((child) => _untrack(child, components_to_register));
  }
  return;
}
function untrack_children_of_invisible_parents(node, components_to_register) {
  if (node.props.shared_props.visible !== true) {
    _untrack(node, components_to_register);
  }
  return node;
}
function mark_component_invisible_if_visible(node, hidden_on_startup) {
  if (node.props.shared_props.visible === true) {
    hidden_on_startup.add(node.id);
    node.props.shared_props.visible = false;
  }
  node.children.forEach((child) => {
    mark_component_invisible_if_visible(child, hidden_on_startup);
  });
  return node;
}
function untrack_children_of_closed_accordions_or_inactive_tabs(node, components_to_register, hidden_on_startup) {
  if (node.type === "accordion" && node.props.props.open === false) {
    _untrack(node, components_to_register);
    if (node.children) {
      node.children.forEach((child) => {
        mark_component_invisible_if_visible(child, hidden_on_startup);
      });
    }
  }
  if (node.type === "tabs") {
    node.children.forEach((child) => {
      if (child.type === "tabitem" && child.props.props.id !== //@ts-ignore
      (node.props.props.selected || node.props.props.initial_tabs[0].id)) {
        _untrack(child, components_to_register);
        mark_component_invisible_if_visible(child, hidden_on_startup);
      }
    });
  }
  return node;
}
function apply_initial_tabs(node, initial_tabs) {
  if (node.type === "tabs" && node.id in initial_tabs) {
    const tabs = initial_tabs[node.id].sort((a, b) => a.order - b.order);
    node.props.props.initial_tabs = tabs;
  } else if (node.type === "tabitem") {
    node.props.props.component_id = node.id;
  }
  return node;
}
function _gather_initial_tabs(node, initial_tabs, parent_tab_id, order) {
  if (parent_tab_id !== null && node.type === "tabitem") {
    if (!(parent_tab_id in initial_tabs)) {
      initial_tabs[parent_tab_id] = [];
    }
    if (!("id" in node.props.props)) {
      node.props.props.id = node.id;
    }
    const i18n = node.props.props.i18n;
    const raw_label = node.props.shared_props.label;
    initial_tabs[parent_tab_id].push({
      label: i18n ? i18n(raw_label) : raw_label,
      id: node.props.props.id,
      elem_id: node.props.shared_props.elem_id,
      visible: node.props.shared_props.visible,
      interactive: node.props.shared_props.interactive,
      scale: node.props.shared_props.scale || null,
      component_id: node.id
    });
    node.props.props.order = order;
  }
  if (node.children) {
    node.children.forEach((child, i) => {
      _gather_initial_tabs(child, initial_tabs, node.type === "tabs" ? node.id : null, node.type === "tabs" ? i : null);
    });
  }
  return;
}
function gather_initial_tabs(node, initial_tabs) {
  function single_visit(node2) {
    if ("children" in node2 && node2.children.length > 0) {
      node2.children?.forEach((child) => _gather_initial_tabs(child, initial_tabs, node2.type === "tabs" ? node2.id : null, null));
    }
  }
  return single_visit(node);
}
function find_node_by_id(tree, id) {
  if (tree.id === id) {
    return tree;
  }
  if (tree.children) {
    for (const child of tree.children) {
      const result = find_node_by_id(child, id);
      if (result) {
        return result;
      }
    }
  }
  return null;
}
function find_parent(tree, id) {
  if (tree.children) {
    for (const child of tree.children) {
      if (child.id === id) {
        return tree;
      }
      const result = find_parent(child, id);
      if (result) {
        return result;
      }
    }
  }
  return null;
}

export { AppTree, Embed };
//# sourceMappingURL=index-DuBh77GF.js.map
