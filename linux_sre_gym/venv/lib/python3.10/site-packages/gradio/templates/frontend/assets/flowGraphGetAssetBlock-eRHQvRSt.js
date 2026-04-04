import { F as FlowGraphBlock, g as getNumericValue, G as GetFlowGraphAssetWithType } from './KHR_interactivity-BPhVhH5L.js';
import { R as RichTypeAny, F as FlowGraphInteger } from './declarationMapper-eK_q7dJF.js';
import { R as RegisterClass } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * A block that will deliver an asset as an output, based on its type and place in the assets index.
 *
 * The assets are loaded from the assetsContext defined in the context running this block. The assetsContext is a class extending AbstractClass,
 * meaning it can be a Scene, an AssetsContainers, and any other class that extends AbstractClass.
 */
class FlowGraphGetAssetBlock extends FlowGraphBlock {
    constructor(
    /**
     * the configuration of the block
     */
    config) {
        super(config);
        this.config = config;
        this.type = this.registerDataInput("type", RichTypeAny, config.type);
        this.value = this.registerDataOutput("value", RichTypeAny);
        this.index = this.registerDataInput("index", RichTypeAny, new FlowGraphInteger(getNumericValue(config.index ?? -1)));
    }
    _updateOutputs(context) {
        const type = this.type.getValue(context);
        const index = this.index.getValue(context);
        // get the asset from the context
        const asset = GetFlowGraphAssetWithType(context.assetsContext, type, getNumericValue(index), this.config.useIndexAsUniqueId);
        this.value.setValue(asset, context);
    }
    /**
     * Gets the class name of this block
     * @returns the class name
     */
    getClassName() {
        return "FlowGraphGetAssetBlock" /* FlowGraphBlockNames.GetAsset */;
    }
}
RegisterClass("FlowGraphGetAssetBlock" /* FlowGraphBlockNames.GetAsset */, FlowGraphGetAssetBlock);

export { FlowGraphGetAssetBlock };
//# sourceMappingURL=flowGraphGetAssetBlock-eRHQvRSt.js.map
