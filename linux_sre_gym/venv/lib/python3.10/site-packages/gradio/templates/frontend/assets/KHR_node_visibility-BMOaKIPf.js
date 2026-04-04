import { aG as unregisterGLTFExtension, aH as registerGLTFExtension } from './index-CxR0Yf_7.js';
import { A as AddObjectAccessorToKey } from './objectModelMapping-DnUo8NMz.js';
import './index-BvBk1Iap.js';

const NAME = "KHR_node_visibility";
// object model extension for visibility
AddObjectAccessorToKey("/nodes/{}/extensions/KHR_node_visibility/visible", {
    get: (node) => {
        const tn = node._babylonTransformNode;
        if (tn && tn.isVisible !== undefined) {
            return tn.isVisible;
        }
        return true;
    },
    set: (value, node) => {
        node._primitiveBabylonMeshes?.forEach((mesh) => {
            mesh.inheritVisibility = true;
        });
        if (node._babylonTransformNode) {
            node._babylonTransformNode.isVisible = value;
        }
        node._primitiveBabylonMeshes?.forEach((mesh) => {
            mesh.isVisible = value;
        });
    },
    getTarget: (node) => node._babylonTransformNode,
    getPropertyName: [() => "isVisible"],
    type: "boolean",
});
/**
 * Loader extension for KHR_node_visibility
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
class KHR_node_visibility {
    /**
     * @internal
     */
    constructor(loader) {
        /**
         * The name of this extension.
         */
        this.name = NAME;
        this._loader = loader;
        this.enabled = loader.isExtensionUsed(NAME);
    }
    onReady() {
        if (!this._loader) {
            return;
        }
        const nodes = this._loader.gltf.nodes;
        if (nodes) {
            for (const node of nodes) {
                const babylonTransformNode = node._babylonTransformNode;
                if (babylonTransformNode) {
                    babylonTransformNode.inheritVisibility = true;
                    if (node.extensions && node.extensions.KHR_node_visibility && node.extensions.KHR_node_visibility.visible === false) {
                        babylonTransformNode.isVisible = false;
                    }
                }
            }
        }
    }
    dispose() {
        delete this._loader;
    }
}
unregisterGLTFExtension(NAME);
registerGLTFExtension(NAME, true, (loader) => new KHR_node_visibility(loader));

export { KHR_node_visibility };
//# sourceMappingURL=KHR_node_visibility-BMOaKIPf.js.map
