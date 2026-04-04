import { F as FlowGraphBlock } from './KHR_interactivity-BPhVhH5L.js';
import { R as RichTypeAny, b as RichTypeNumber } from './declarationMapper-eK_q7dJF.js';
import { R as RegisterClass } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * A block that outputs elements from the context
 */
class FlowGraphContextBlock extends FlowGraphBlock {
    constructor(config) {
        super(config);
        this.userVariables = this.registerDataOutput("userVariables", RichTypeAny);
        this.executionId = this.registerDataOutput("executionId", RichTypeNumber);
    }
    _updateOutputs(context) {
        this.userVariables.setValue(context.userVariables, context);
        this.executionId.setValue(context.executionId, context);
    }
    serialize(serializationObject) {
        super.serialize(serializationObject);
    }
    getClassName() {
        return "FlowGraphContextBlock" /* FlowGraphBlockNames.Context */;
    }
}
RegisterClass("FlowGraphContextBlock" /* FlowGraphBlockNames.Context */, FlowGraphContextBlock);

export { FlowGraphContextBlock };
//# sourceMappingURL=flowGraphContextBlock-D4tc8TQP.js.map
