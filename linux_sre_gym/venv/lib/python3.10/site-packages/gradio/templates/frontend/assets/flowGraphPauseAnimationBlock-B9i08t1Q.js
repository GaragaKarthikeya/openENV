import { b as FlowGraphExecutionBlockWithOutSignal } from './KHR_interactivity-BPhVhH5L.js';
import { R as RichTypeAny } from './declarationMapper-eK_q7dJF.js';
import { R as RegisterClass } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * @experimental
 * Block that pauses a running animation
 */
class FlowGraphPauseAnimationBlock extends FlowGraphExecutionBlockWithOutSignal {
    constructor(config) {
        super(config);
        this.animationToPause = this.registerDataInput("animationToPause", RichTypeAny);
    }
    _execute(context) {
        const animationToPauseValue = this.animationToPause.getValue(context);
        animationToPauseValue.pause();
        this.out._activateSignal(context);
    }
    /**
     * @returns class name of the block.
     */
    getClassName() {
        return "FlowGraphPauseAnimationBlock" /* FlowGraphBlockNames.PauseAnimation */;
    }
}
RegisterClass("FlowGraphPauseAnimationBlock" /* FlowGraphBlockNames.PauseAnimation */, FlowGraphPauseAnimationBlock);

export { FlowGraphPauseAnimationBlock };
//# sourceMappingURL=flowGraphPauseAnimationBlock-B9i08t1Q.js.map
