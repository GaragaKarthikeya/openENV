import { s as styles_default, c as classRenderer_v3_unified_default, a as classDiagram_default, C as ClassDB } from './chunk-B4BG7PRW-DsvgzSbP.js';
import { _ as __name } from './mermaid.core-m0ZQP3yq.js';
import './chunk-FMBD7UC4-hWqaK4zc.js';
import './chunk-55IACEB6-DHcY6oTe.js';
import './select-k8gDf_61.js';
import './chunk-QN33PNHL-Bjd8zJAX.js';
import './index-BvBk1Iap.js';
import './i18n-CMo5lpzy.js';
import './step-TZOpqHBK.js';
import './dispatch-tQmgj1It.js';

// src/diagrams/class/classDiagram-v2.ts
var diagram = {
  parser: classDiagram_default,
  get db() {
    return new ClassDB();
  },
  renderer: classRenderer_v3_unified_default,
  styles: styles_default,
  init: /* @__PURE__ */ __name((cnf) => {
    if (!cnf.class) {
      cnf.class = {};
    }
    cnf.class.arrowMarkerAbsolute = cnf.arrowMarkerAbsolute;
  }, "init")
};

export { diagram };
//# sourceMappingURL=classDiagram-v2-WZHVMYZB-CW44iY6c.js.map
