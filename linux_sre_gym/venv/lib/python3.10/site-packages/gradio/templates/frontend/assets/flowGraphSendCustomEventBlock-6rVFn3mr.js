import { b as FlowGraphExecutionBlockWithOutSignal } from './KHR_interactivity-BPhVhH5L.js';
import { R as RegisterClass } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './declarationMapper-eK_q7dJF.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * A block that sends a custom event.
 * To receive this event you need to use the ReceiveCustomEvent block.
 * This block has no output, but does have inputs based on the eventData from the configuration.
 * @see FlowGraphReceiveCustomEventBlock
 */
class FlowGraphSendCustomEventBlock extends FlowGraphExecutionBlockWithOutSignal {
    constructor(
    /**
     * the configuration of the block
     */
    config) {
        super(config);
        this.config = config;
        for (const key in this.config.eventData) {
            this.registerDataInput(key, this.config.eventData[key].type, this.config.eventData[key].value);
        }
    }
    _execute(context) {
        const eventId = this.config.eventId;
        // eventData is a map with the key being the data input's name, and value being the data input's value
        const eventData = {};
        for (const port of this.dataInputs) {
            eventData[port.name] = port.getValue(context);
        }
        context.configuration.coordinator.notifyCustomEvent(eventId, eventData);
        this.out._activateSignal(context);
    }
    /**
     * @returns class name of the block.
     */
    getClassName() {
        return "FlowGraphReceiveCustomEventBlock" /* FlowGraphBlockNames.ReceiveCustomEvent */;
    }
}
RegisterClass("FlowGraphReceiveCustomEventBlock" /* FlowGraphBlockNames.ReceiveCustomEvent */, FlowGraphSendCustomEventBlock);

export { FlowGraphSendCustomEventBlock };
//# sourceMappingURL=flowGraphSendCustomEventBlock-6rVFn3mr.js.map
