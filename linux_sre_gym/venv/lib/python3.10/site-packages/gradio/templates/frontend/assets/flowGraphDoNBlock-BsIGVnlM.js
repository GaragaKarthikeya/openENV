import { F as FlowGraphInteger, i as RichTypeFlowGraphInteger } from './declarationMapper-eK_q7dJF.js';
import { b as FlowGraphExecutionBlockWithOutSignal } from './KHR_interactivity-BPhVhH5L.js';
import { R as RegisterClass } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * A block that executes a branch a set number of times.
 */
class FlowGraphDoNBlock extends FlowGraphExecutionBlockWithOutSignal {
    constructor(
    /**
     * [Object] the configuration of the block
     */
    config = {}) {
        super(config);
        this.config = config;
        this.config.startIndex = config.startIndex ?? new FlowGraphInteger(0);
        this.reset = this._registerSignalInput("reset");
        this.maxExecutions = this.registerDataInput("maxExecutions", RichTypeFlowGraphInteger);
        this.executionCount = this.registerDataOutput("executionCount", RichTypeFlowGraphInteger, new FlowGraphInteger(0));
    }
    _execute(context, callingSignal) {
        if (callingSignal === this.reset) {
            this.executionCount.setValue(this.config.startIndex, context);
        }
        else {
            const currentCountValue = this.executionCount.getValue(context);
            if (currentCountValue.value < this.maxExecutions.getValue(context).value) {
                this.executionCount.setValue(new FlowGraphInteger(currentCountValue.value + 1), context);
                this.out._activateSignal(context);
            }
        }
    }
    /**
     * @returns class name of the block.
     */
    getClassName() {
        return "FlowGraphDoNBlock" /* FlowGraphBlockNames.DoN */;
    }
}
RegisterClass("FlowGraphDoNBlock" /* FlowGraphBlockNames.DoN */, FlowGraphDoNBlock);

export { FlowGraphDoNBlock };
//# sourceMappingURL=flowGraphDoNBlock-BsIGVnlM.js.map
