import { F as FlowGraphBlock, f as defaultValueSerializationFunction } from './KHR_interactivity-BPhVhH5L.js';
import { j as getRichTypeFromValue } from './declarationMapper-eK_q7dJF.js';
import { R as RegisterClass } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * Block that returns a constant value.
 */
class FlowGraphConstantBlock extends FlowGraphBlock {
    constructor(
    /**
     * the configuration of the block
     */
    config) {
        super(config);
        this.config = config;
        this.output = this.registerDataOutput("output", getRichTypeFromValue(config.value));
    }
    _updateOutputs(context) {
        this.output.setValue(this.config.value, context);
    }
    /**
     * Gets the class name of this block
     * @returns the class name
     */
    getClassName() {
        return "FlowGraphConstantBlock" /* FlowGraphBlockNames.Constant */;
    }
    /**
     * Serializes this block
     * @param serializationObject the object to serialize to
     * @param valueSerializeFunction the function to use to serialize the value
     */
    serialize(serializationObject = {}, valueSerializeFunction = defaultValueSerializationFunction) {
        super.serialize(serializationObject);
        valueSerializeFunction("value", this.config.value, serializationObject.config);
    }
}
RegisterClass("FlowGraphConstantBlock" /* FlowGraphBlockNames.Constant */, FlowGraphConstantBlock);

export { FlowGraphConstantBlock };
//# sourceMappingURL=flowGraphConstantBlock-Cd6_VUbG.js.map
