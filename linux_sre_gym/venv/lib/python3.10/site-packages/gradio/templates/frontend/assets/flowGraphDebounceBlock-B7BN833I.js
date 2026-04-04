import { b as RichTypeNumber } from './declarationMapper-eK_q7dJF.js';
import { b as FlowGraphExecutionBlockWithOutSignal } from './KHR_interactivity-BPhVhH5L.js';
import { R as RegisterClass } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * This block debounces the execution of a input, i.e. ensures that the input is only executed once every X times
 */
class FlowGraphDebounceBlock extends FlowGraphExecutionBlockWithOutSignal {
    constructor(config) {
        super(config);
        this.count = this.registerDataInput("count", RichTypeNumber);
        this.reset = this._registerSignalInput("reset");
        this.currentCount = this.registerDataOutput("currentCount", RichTypeNumber);
    }
    _execute(context, callingSignal) {
        if (callingSignal === this.reset) {
            context._setExecutionVariable(this, "debounceCount", 0);
            return;
        }
        const count = this.count.getValue(context);
        const currentCount = context._getExecutionVariable(this, "debounceCount", 0);
        const newCount = currentCount + 1;
        this.currentCount.setValue(newCount, context);
        context._setExecutionVariable(this, "debounceCount", newCount);
        if (newCount >= count) {
            this.out._activateSignal(context);
            context._setExecutionVariable(this, "debounceCount", 0);
        }
    }
    /**
     * @returns class name of the block.
     */
    getClassName() {
        return "FlowGraphDebounceBlock" /* FlowGraphBlockNames.Debounce */;
    }
}
RegisterClass("FlowGraphDebounceBlock" /* FlowGraphBlockNames.Debounce */, FlowGraphDebounceBlock);

export { FlowGraphDebounceBlock };
//# sourceMappingURL=flowGraphDebounceBlock-B7BN833I.js.map
