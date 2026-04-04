import { F as FlowGraphBlock } from './KHR_interactivity-BPhVhH5L.js';
import { R as RichTypeAny } from './declarationMapper-eK_q7dJF.js';
import './index-BvBk1Iap.js';
import './index-CxR0Yf_7.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * a glTF-based FlowGraph block that provides arrays with babylon object, based on the glTF tree
 * Can be used, for example, to get animation index from a glTF animation
 */
class FlowGraphGLTFDataProvider extends FlowGraphBlock {
    constructor(config) {
        super();
        const glTF = config.glTF;
        const animationGroups = glTF.animations?.map((a) => a._babylonAnimationGroup) || [];
        this.animationGroups = this.registerDataOutput("animationGroups", RichTypeAny, animationGroups);
        const nodes = glTF.nodes?.map((n) => n._babylonTransformNode) || [];
        this.nodes = this.registerDataOutput("nodes", RichTypeAny, nodes);
    }
    getClassName() {
        return "FlowGraphGLTFDataProvider";
    }
}

export { FlowGraphGLTFDataProvider };
//# sourceMappingURL=flowGraphGLTFDataProvider-BF36Cs2j.js.map
