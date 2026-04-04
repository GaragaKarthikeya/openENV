import { R as RegisterClass } from './index-CxR0Yf_7.js';
import { b as FlowGraphExecutionBlockWithOutSignal, g as getNumericValue } from './KHR_interactivity-BPhVhH5L.js';
import { i as RichTypeFlowGraphInteger } from './declarationMapper-eK_q7dJF.js';
import './index-BvBk1Iap.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * This block cancels a delay that was previously scheduled.
 */
class FlowGraphCancelDelayBlock extends FlowGraphExecutionBlockWithOutSignal {
    constructor(config) {
        super(config);
        this.delayIndex = this.registerDataInput("delayIndex", RichTypeFlowGraphInteger);
    }
    _execute(context, _callingSignal) {
        const delayIndex = getNumericValue(this.delayIndex.getValue(context));
        if (delayIndex <= 0 || isNaN(delayIndex) || !isFinite(delayIndex)) {
            return this._reportError(context, "Invalid delay index");
        }
        const timers = context._getGlobalContextVariable("pendingDelays", []);
        const timer = timers[delayIndex];
        if (timer) {
            timer.dispose();
            // not removing it from the array. Disposing it will clear all of its resources
        }
        // activate the out output flow
        this.out._activateSignal(context);
    }
    getClassName() {
        return "FlowGraphCancelDelayBlock" /* FlowGraphBlockNames.CancelDelay */;
    }
}
RegisterClass("FlowGraphCancelDelayBlock" /* FlowGraphBlockNames.CancelDelay */, FlowGraphCancelDelayBlock);

export { FlowGraphCancelDelayBlock };
//# sourceMappingURL=flowGraphCancelDelayBlock-DdnCJIHp.js.map
