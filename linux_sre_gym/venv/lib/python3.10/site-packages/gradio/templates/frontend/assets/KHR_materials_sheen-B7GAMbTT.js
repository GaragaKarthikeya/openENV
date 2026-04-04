import { GLTFLoader } from './glTFLoader-BbG05oe0.js';
import { C as Color3, aG as unregisterGLTFExtension, aH as registerGLTFExtension } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './bone-C6xIrJX4.js';
import './skeleton-DzANFt7R.js';
import './rawTexture-Bto4WZO2.js';
import './assetContainer-7GPvH2S_.js';
import './objectModelMapping-DnUo8NMz.js';

const NAME = "KHR_materials_sheen";
/**
 * [Specification](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_sheen/README.md)
 * [Playground Sample](https://www.babylonjs-playground.com/frame.html#BNIZX6#4)
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
class KHR_materials_sheen {
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
        this.order = 190;
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
            const promises = new Array();
            promises.push(this._loader.loadMaterialPropertiesAsync(context, material, babylonMaterial));
            promises.push(this._loadSheenPropertiesAsync(extensionContext, extension, babylonMaterial));
            // eslint-disable-next-line github/no-then
            return await Promise.all(promises).then(() => { });
        });
    }
    // eslint-disable-next-line @typescript-eslint/promise-function-async, no-restricted-syntax
    _loadSheenPropertiesAsync(context, properties, babylonMaterial) {
        const adapter = this._loader._getOrCreateMaterialAdapter(babylonMaterial);
        const promises = new Array();
        adapter.configureFuzz();
        // Set non-texture properties immediately
        const sheenColor = properties.sheenColorFactor !== undefined ? Color3.FromArray(properties.sheenColorFactor) : Color3.Black();
        const sheenRoughness = properties.sheenRoughnessFactor !== undefined ? properties.sheenRoughnessFactor : 0.0;
        adapter.fuzzWeight = 1; // KHR_materials_sheen assumes intensity of 1
        adapter.fuzzColor = sheenColor;
        adapter.fuzzRoughness = sheenRoughness;
        // Load textures
        if (properties.sheenColorTexture) {
            promises.push(this._loader.loadTextureInfoAsync(`${context}/sheenColorTexture`, properties.sheenColorTexture, (texture) => {
                texture.name = `${babylonMaterial.name} (Sheen Color)`;
                adapter.fuzzColorTexture = texture;
            }));
        }
        if (properties.sheenRoughnessTexture) {
            properties.sheenRoughnessTexture.nonColorData = true;
            promises.push(this._loader.loadTextureInfoAsync(`${context}/sheenRoughnessTexture`, properties.sheenRoughnessTexture, (texture) => {
                texture.name = `${babylonMaterial.name} (Sheen Roughness)`;
                adapter.fuzzRoughnessTexture = texture;
            }));
        }
        // eslint-disable-next-line github/no-then
        return Promise.all(promises).then(() => { });
    }
}
unregisterGLTFExtension(NAME);
registerGLTFExtension(NAME, true, (loader) => new KHR_materials_sheen(loader));

export { KHR_materials_sheen };
//# sourceMappingURL=KHR_materials_sheen-B7GAMbTT.js.map
