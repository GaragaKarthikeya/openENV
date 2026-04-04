import { c as FlowGraphEventBlock } from './KHR_interactivity-BPhVhH5L.js';
import { R as RegisterClass } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './declarationMapper-eK_q7dJF.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * Block that triggers when a scene is ready.
 */
class FlowGraphSceneReadyEventBlock extends FlowGraphEventBlock {
    constructor() {
        super(...arguments);
        this.initPriority = -1;
        this.type = "SceneReady" /* FlowGraphEventType.SceneReady */;
    }
    _executeEvent(context, _payload) {
        this._execute(context);
        return true;
    }
    _preparePendingTasks(context) {
        // no-op
    }
    _cancelPendingTasks(context) {
        // no-op
    }
    /**
     * @returns class name of the block.
     */
    getClassName() {
        return "FlowGraphSceneReadyEventBlock" /* FlowGraphBlockNames.SceneReadyEvent */;
    }
}
RegisterClass("FlowGraphSceneReadyEventBlock" /* FlowGraphBlockNames.SceneReadyEvent */, FlowGraphSceneReadyEventBlock);

export { FlowGraphSceneReadyEventBlock };
//# sourceMappingURL=flowGraphSceneReadyEventBlock-vtf4yMOr.js.map
