import { GLTFLoader, ArrayItem } from './glTFLoader-BbG05oe0.js';
import { aG as unregisterGLTFExtension, aH as registerGLTFExtension } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './bone-C6xIrJX4.js';
import './skeleton-DzANFt7R.js';
import './rawTexture-Bto4WZO2.js';
import './assetContainer-7GPvH2S_.js';
import './objectModelMapping-DnUo8NMz.js';

const NAME = "EXT_texture_avif";
/**
 * [glTF PR](https://github.com/KhronosGroup/glTF/pull/2235)
 * [Specification](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_texture_avif/README.md)
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
class EXT_texture_avif {
    /**
     * @internal
     */
    constructor(loader) {
        /** The name of this extension. */
        this.name = NAME;
        this._loader = loader;
        this.enabled = loader.isExtensionUsed(NAME);
    }
    /** @internal */
    dispose() {
        this._loader = null;
    }
    /**
     * @internal
     */
    // eslint-disable-next-line no-restricted-syntax
    _loadTextureAsync(context, texture, assign) {
        return GLTFLoader.LoadExtensionAsync(context, texture, this.name, async (extensionContext, extension) => {
            const sampler = texture.sampler == undefined ? GLTFLoader.DefaultSampler : ArrayItem.Get(`${context}/sampler`, this._loader.gltf.samplers, texture.sampler);
            const image = ArrayItem.Get(`${extensionContext}/source`, this._loader.gltf.images, extension.source);
            return await this._loader._createTextureAsync(context, sampler, image, (babylonTexture) => {
                assign(babylonTexture);
            }, undefined, !texture._textureInfo.nonColorData);
        });
    }
}
unregisterGLTFExtension(NAME);
registerGLTFExtension(NAME, true, (loader) => new EXT_texture_avif(loader));

export { EXT_texture_avif };
//# sourceMappingURL=EXT_texture_avif-CYS9TPkR.js.map
