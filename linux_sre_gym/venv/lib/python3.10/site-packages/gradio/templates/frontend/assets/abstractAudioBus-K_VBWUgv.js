import { A as AbstractAudioOutNode } from './webAudioBaseSubGraph-CuchFrSM.js';

/**
 * Abstract class representing an audio bus with volume control.
 *
 * An audio bus is a node in the audio graph that can have multiple inputs and outputs. It is typically used to group
 * sounds together and apply effects to them.
 */
class AbstractAudioBus extends AbstractAudioOutNode {
    constructor(name, engine) {
        super(name, engine, 3 /* AudioNodeType.HAS_INPUTS_AND_OUTPUTS */);
    }
}

export { AbstractAudioBus as A };
//# sourceMappingURL=abstractAudioBus-K_VBWUgv.js.map
