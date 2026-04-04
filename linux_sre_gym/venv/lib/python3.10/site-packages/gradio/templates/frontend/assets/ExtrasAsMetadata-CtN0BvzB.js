import { aG as unregisterGLTFExtension, aH as registerGLTFExtension } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';

const NAME = "ExtrasAsMetadata";
/**
 * Store glTF extras (if present) in BJS objects' metadata
 */
class ExtrasAsMetadata {
    _assignExtras(babylonObject, gltfProp) {
        if (gltfProp.extras && Object.keys(gltfProp.extras).length > 0) {
            const metadata = (babylonObject.metadata = babylonObject.metadata || {});
            const gltf = (metadata.gltf = metadata.gltf || {});
            gltf.extras = gltfProp.extras;
        }
    }
    /**
     * @internal
     */
    constructor(loader) {
        /**
         * The name of this extension.
         */
        this.name = NAME;
        /**
         * Defines whether this extension is enabled.
         */
        this.enabled = true;
        this._loader = loader;
    }
    /** @internal */
    dispose() {
        this._loader = null;
    }
    /**
     * @internal
     */
    // eslint-disable-next-line no-restricted-syntax
    loadNodeAsync(context, node, assign) {
        return this._loader.loadNodeAsync(context, node, (babylonTransformNode) => {
            this._assignExtras(babylonTransformNode, node);
            assign(babylonTransformNode);
        });
    }
    /**
     * @internal
     */
    // eslint-disable-next-line no-restricted-syntax
    loadCameraAsync(context, camera, assign) {
        return this._loader.loadCameraAsync(context, camera, (babylonCamera) => {
            this._assignExtras(babylonCamera, camera);
            assign(babylonCamera);
        });
    }
    /**
     * @internal
     */
    createMaterial(context, material, babylonDrawMode) {
        const babylonMaterial = this._loader.createMaterial(context, material, babylonDrawMode);
        this._assignExtras(babylonMaterial, material);
        return babylonMaterial;
    }
    /**
     * @internal
     */
    // eslint-disable-next-line no-restricted-syntax
    loadAnimationAsync(context, animation) {
        // eslint-disable-next-line github/no-then
        return this._loader.loadAnimationAsync(context, animation).then((babylonAnimation) => {
            this._assignExtras(babylonAnimation, animation);
            return babylonAnimation;
        });
    }
}
unregisterGLTFExtension(NAME);
registerGLTFExtension(NAME, false, (loader) => new ExtrasAsMetadata(loader));

export { ExtrasAsMetadata };
//# sourceMappingURL=ExtrasAsMetadata-CtN0BvzB.js.map
