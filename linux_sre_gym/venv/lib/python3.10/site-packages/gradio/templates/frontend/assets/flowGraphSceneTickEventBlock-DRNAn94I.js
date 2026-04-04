import { c as FlowGraphEventBlock } from './KHR_interactivity-BPhVhH5L.js';
import { R as RegisterClass } from './index-CxR0Yf_7.js';
import { b as RichTypeNumber } from './declarationMapper-eK_q7dJF.js';
import './index-BvBk1Iap.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * Block that triggers on scene tick (before each render).
 */
class FlowGraphSceneTickEventBlock extends FlowGraphEventBlock {
    constructor() {
        super();
        this.type = "SceneBeforeRender" /* FlowGraphEventType.SceneBeforeRender */;
        this.timeSinceStart = this.registerDataOutput("timeSinceStart", RichTypeNumber);
        this.deltaTime = this.registerDataOutput("deltaTime", RichTypeNumber);
    }
    /**
     * @internal
     */
    _preparePendingTasks(_context) {
        // no-op
    }
    /**
     * @internal
     */
    _executeEvent(context, payload) {
        this.timeSinceStart.setValue(payload.timeSinceStart, context);
        this.deltaTime.setValue(payload.deltaTime, context);
        this._execute(context);
        return true;
    }
    /**
     * @internal
     */
    _cancelPendingTasks(_context) {
        // no-op
    }
    /**
     * @returns class name of the block.
     */
    getClassName() {
        return "FlowGraphSceneTickEventBlock" /* FlowGraphBlockNames.SceneTickEvent */;
    }
}
RegisterClass("FlowGraphSceneTickEventBlock" /* FlowGraphBlockNames.SceneTickEvent */, FlowGraphSceneTickEventBlock);

export { FlowGraphSceneTickEventBlock };
//# sourceMappingURL=flowGraphSceneTickEventBlock-DRNAn94I.js.map
