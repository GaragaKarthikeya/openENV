import { F as FlowGraphBlock, g as getNumericValue } from './KHR_interactivity-BPhVhH5L.js';
import { R as RichTypeAny, F as FlowGraphInteger } from './declarationMapper-eK_q7dJF.js';
import { R as RegisterClass } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * This simple Util block takes an array as input and selects a single element from it.
 */
class FlowGraphArrayIndexBlock extends FlowGraphBlock {
    /**
     * Construct a FlowGraphArrayIndexBlock.
     * @param config construction parameters
     */
    constructor(config) {
        super(config);
        this.config = config;
        this.array = this.registerDataInput("array", RichTypeAny);
        this.index = this.registerDataInput("index", RichTypeAny, new FlowGraphInteger(-1));
        this.value = this.registerDataOutput("value", RichTypeAny);
    }
    /**
     * @internal
     */
    _updateOutputs(context) {
        const array = this.array.getValue(context);
        const index = getNumericValue(this.index.getValue(context));
        if (array && index >= 0 && index < array.length) {
            this.value.setValue(array[index], context);
        }
        else {
            this.value.setValue(null, context);
        }
    }
    /**
     * Serializes this block
     * @param serializationObject the object to serialize to
     */
    serialize(serializationObject) {
        super.serialize(serializationObject);
    }
    getClassName() {
        return "FlowGraphArrayIndexBlock" /* FlowGraphBlockNames.ArrayIndex */;
    }
}
RegisterClass("FlowGraphArrayIndexBlock" /* FlowGraphBlockNames.ArrayIndex */, FlowGraphArrayIndexBlock);

export { FlowGraphArrayIndexBlock };
//# sourceMappingURL=flowGraphArrayIndexBlock-NJcLU5dO.js.map
