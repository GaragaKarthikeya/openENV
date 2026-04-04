import { F as FlowGraphBlock } from './KHR_interactivity-BPhVhH5L.js';
import { q as RichTypeString, R as RichTypeAny } from './declarationMapper-eK_q7dJF.js';
import { R as RegisterClass } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * A flow graph block that takes a function name, an object and an optional context as inputs and calls the function on the object.
 */
class FlowGraphFunctionReferenceBlock extends FlowGraphBlock {
    constructor(
    /**
     * the configuration of the block
     */
    config) {
        super(config);
        this.functionName = this.registerDataInput("functionName", RichTypeString);
        this.object = this.registerDataInput("object", RichTypeAny);
        this.context = this.registerDataInput("context", RichTypeAny, null);
        this.output = this.registerDataOutput("output", RichTypeAny);
    }
    _updateOutputs(context) {
        const functionName = this.functionName.getValue(context);
        const object = this.object.getValue(context);
        const contextValue = this.context.getValue(context);
        if (object && functionName) {
            const func = object[functionName];
            if (func && typeof func === "function") {
                this.output.setValue(func.bind(contextValue), context);
            }
        }
    }
    getClassName() {
        return "FlowGraphFunctionReference" /* FlowGraphBlockNames.FunctionReference */;
    }
}
RegisterClass("FlowGraphFunctionReference" /* FlowGraphBlockNames.FunctionReference */, FlowGraphFunctionReferenceBlock);

export { FlowGraphFunctionReferenceBlock };
//# sourceMappingURL=flowGraphFunctionReferenceBlock-ByGpsFPE.js.map
