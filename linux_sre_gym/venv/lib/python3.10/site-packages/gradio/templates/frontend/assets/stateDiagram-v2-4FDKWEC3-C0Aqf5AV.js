import { s as styles_default, b as stateRenderer_v3_unified_default, a as stateDiagram_default, S as StateDB } from './chunk-DI55MBZ5-CANvPCpq.js';
import { _ as __name } from './mermaid.core-m0ZQP3yq.js';
import './chunk-55IACEB6-DHcY6oTe.js';
import './select-k8gDf_61.js';
import './chunk-QN33PNHL-Bjd8zJAX.js';
import './index-BvBk1Iap.js';
import './i18n-CMo5lpzy.js';
import './step-TZOpqHBK.js';
import './dispatch-tQmgj1It.js';

// src/diagrams/state/stateDiagram-v2.ts
var diagram = {
  parser: stateDiagram_default,
  get db() {
    return new StateDB(2);
  },
  renderer: stateRenderer_v3_unified_default,
  styles: styles_default,
  init: /* @__PURE__ */ __name((cnf) => {
    if (!cnf.state) {
      cnf.state = {};
    }
    cnf.state.arrowMarkerAbsolute = cnf.arrowMarkerAbsolute;
  }, "init")
};

export { diagram };
//# sourceMappingURL=stateDiagram-v2-4FDKWEC3-C0Aqf5AV.js.map
