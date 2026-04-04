import { GLTFLoader } from './glTFLoader-BbG05oe0.js';
import { C as Color3, aG as unregisterGLTFExtension, aH as registerGLTFExtension } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './bone-C6xIrJX4.js';
import './skeleton-DzANFt7R.js';
import './rawTexture-Bto4WZO2.js';
import './assetContainer-7GPvH2S_.js';
import './objectModelMapping-DnUo8NMz.js';

const NAME = "KHR_materials_diffuse_transmission";
/**
 * [Proposed Specification](https://github.com/KhronosGroup/glTF/pull/1825)
 * !!! Experimental Extension Subject to Changes !!!
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
class KHR_materials_diffuse_transmission {
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
        this.order = 174;
        this._loader = loader;
        this.enabled = this._loader.isExtensionUsed(NAME);
        if (this.enabled) {
            loader.parent.transparencyAsCoverage = true;
        }
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
            promises.push(this._loadTranslucentPropertiesAsync(extensionContext, material, babylonMaterial, extension));
            return await Promise.all(promises).then(() => { });
        });
    }
    // eslint-disable-next-line no-restricted-syntax, @typescript-eslint/promise-function-async
    _loadTranslucentPropertiesAsync(context, material, babylonMaterial, extension) {
        const adapter = this._loader._getOrCreateMaterialAdapter(babylonMaterial);
        adapter.configureSubsurface();
        adapter.subsurfaceWeight = extension.diffuseTransmissionFactor ?? 0;
        adapter.subsurfaceColor = extension.diffuseTransmissionColorFactor !== undefined ? Color3.FromArray(extension.diffuseTransmissionColorFactor) : Color3.White();
        const promises = new Array();
        if (extension.diffuseTransmissionTexture) {
            extension.diffuseTransmissionTexture.nonColorData = true;
            promises.push(this._loader.loadTextureInfoAsync(`${context}/diffuseTransmissionTexture`, extension.diffuseTransmissionTexture).then((texture) => {
                texture.name = `${babylonMaterial.name} (Diffuse Transmission)`;
                adapter.subsurfaceWeightTexture = texture;
            }));
        }
        if (extension.diffuseTransmissionColorTexture) {
            promises.push(this._loader.loadTextureInfoAsync(`${context}/diffuseTransmissionColorTexture`, extension.diffuseTransmissionColorTexture).then((texture) => {
                texture.name = `${babylonMaterial.name} (Diffuse Transmission Color)`;
                adapter.subsurfaceColorTexture = texture;
            }));
        }
        return Promise.all(promises).then(() => { });
    }
}
unregisterGLTFExtension(NAME);
registerGLTFExtension(NAME, true, (loader) => new KHR_materials_diffuse_transmission(loader));

export { KHR_materials_diffuse_transmission };
//# sourceMappingURL=KHR_materials_diffuse_transmission-Bsz9QBT8.js.map
