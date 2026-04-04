import { GLTFLoader } from './glTFLoader-BbG05oe0.js';
import { aG as unregisterGLTFExtension, aH as registerGLTFExtension } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './bone-C6xIrJX4.js';
import './skeleton-DzANFt7R.js';
import './rawTexture-Bto4WZO2.js';
import './assetContainer-7GPvH2S_.js';
import './objectModelMapping-DnUo8NMz.js';

const NAME = "KHR_materials_emissive_strength";
/**
 * [Specification](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_emissive_strength/README.md)
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
class KHR_materials_emissive_strength {
    /**
     * @internal
     */
    constructor(loader) {
        /**
         * The name of this extension.
         */
        this.name = NAME;
        /**
         * Defines a number that determines the order the extensions are applied.
         */
        this.order = 170;
        this._loader = loader;
        this.enabled = this._loader.isExtensionUsed(NAME);
    }
    /** @internal */
    dispose() {
        this._loader = null;
    }
    /**
     * @internal
     */
    // eslint-disable-next-line no-restricted-syntax
    loadMaterialPropertiesAsync(context, material, babylonMaterial) {
        return GLTFLoader.LoadExtensionAsync(context, material, this.name, async (extensionContext, extension) => {
            await this._loader.loadMaterialPropertiesAsync(context, material, babylonMaterial);
            this._loadEmissiveProperties(extensionContext, extension, babylonMaterial);
            return await Promise.resolve();
        });
    }
    _loadEmissiveProperties(context, properties, babylonMaterial) {
        if (properties.emissiveStrength !== undefined) {
            const adapter = this._loader._getOrCreateMaterialAdapter(babylonMaterial);
            adapter.emissionLuminance = properties.emissiveStrength;
        }
    }
}
unregisterGLTFExtension(NAME);
registerGLTFExtension(NAME, true, (loader) => new KHR_materials_emissive_strength(loader));

export { KHR_materials_emissive_strength };
//# sourceMappingURL=KHR_materials_emissive_strength-BlmYyOhP.js.map
