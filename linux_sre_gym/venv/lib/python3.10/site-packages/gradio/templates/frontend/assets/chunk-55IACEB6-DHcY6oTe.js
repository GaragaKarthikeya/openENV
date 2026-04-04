import { _ as __name } from './mermaid.core-m0ZQP3yq.js';
import { s as select } from './select-k8gDf_61.js';

var getDiagramElement = /* @__PURE__ */ __name((id, securityLevel) => {
  let sandboxElement;
  if (securityLevel === "sandbox") {
    sandboxElement = select("#i" + id);
  }
  const root = securityLevel === "sandbox" ? select(sandboxElement.nodes()[0].contentDocument.body) : select("body");
  const svg = root.select(`[id="${id}"]`);
  return svg;
}, "getDiagramElement");

export { getDiagramElement as g };
//# sourceMappingURL=chunk-55IACEB6-DHcY6oTe.js.map
