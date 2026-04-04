import { _ as __name, l as log, I as selectSvgElement, d as configureSvgSize, K as package_default } from './mermaid.core-m0ZQP3yq.js';
import { p as parse } from './treemap-KMMF4GRG-DbDz9Vki.js';
import './index-BvBk1Iap.js';
import './i18n-CMo5lpzy.js';
import './step-TZOpqHBK.js';
import './dispatch-tQmgj1It.js';
import './select-k8gDf_61.js';
import './min-C2QU5z8n.js';
import './_baseUniq-DxAWy4wm.js';

var parser = {
  parse: /* @__PURE__ */ __name(async (input) => {
    const ast = await parse("info", input);
    log.debug(ast);
  }, "parse")
};

// src/diagrams/info/infoDb.ts
var DEFAULT_INFO_DB = {
  version: package_default.version + ("" )
};
var getVersion = /* @__PURE__ */ __name(() => DEFAULT_INFO_DB.version, "getVersion");
var db = {
  getVersion
};

// src/diagrams/info/infoRenderer.ts
var draw = /* @__PURE__ */ __name((text, id, version) => {
  log.debug("rendering info diagram\n" + text);
  const svg = selectSvgElement(id);
  configureSvgSize(svg, 100, 400, true);
  const group = svg.append("g");
  group.append("text").attr("x", 100).attr("y", 40).attr("class", "version").attr("font-size", 32).style("text-anchor", "middle").text(`v${version}`);
}, "draw");
var renderer = { draw };

// src/diagrams/info/infoDiagram.ts
var diagram = {
  parser,
  db,
  renderer
};

export { diagram };
//# sourceMappingURL=infoDiagram-WHAUD3N6-BnuU1s7r.js.map
