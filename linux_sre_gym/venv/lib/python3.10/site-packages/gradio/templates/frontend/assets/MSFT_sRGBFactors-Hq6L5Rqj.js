import { GLTFLoader } from './glTFLoader-BbG05oe0.js';
import { aG as unregisterGLTFExtension, aH as registerGLTFExtension } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './bone-C6xIrJX4.js';
import './skeleton-DzANFt7R.js';
import './rawTexture-Bto4WZO2.js';
import './assetContainer-7GPvH2S_.js';
import './objectModelMapping-DnUo8NMz.js';

const NAME = "MSFT_sRGBFactors";
/** @internal */
// eslint-disable-next-line @typescript-eslint/naming-convention
class MSFT_sRGBFactors {
    /** @internal */
    constructor(loader) {
        /** @internal */
        this.name = NAME;
        this._loader = loader;
        this.enabled = this._loader.isExtensionUsed(NAME);
    }
    /** @internal */
    dispose() {
        this._loader = null;
    }
    /** @internal*/
    // eslint-disable-next-line no-restricted-syntax
    loadMaterialPropertiesAsync(context, material, babylonMaterial) {
        return GLTFLoader.LoadExtraAsync(context, material, this.name, async (extraContext, extra) => {
            if (extra) {
                const adapter = this._loader._getOrCreateMaterialAdapter(babylonMaterial);
                const promise = this._loader.loadMaterialPropertiesAsync(context, material, babylonMaterial);
                const useExactSrgbConversions = babylonMaterial.getScene().getEngine().useExactSrgbConversions;
                if (!adapter.baseColorTexture) {
                    adapter.baseColor.toLinearSpaceToRef(adapter.baseColor, useExactSrgbConversions);
                }
                if (!adapter.specularColorTexture) {
                    adapter.specularColor.toLinearSpaceToRef(adapter.specularColor, useExactSrgbConversions);
                }
                return await promise;
            }
        });
    }
}
unregisterGLTFExtension(NAME);
registerGLTFExtension(NAME, true, (loader) => new MSFT_sRGBFactors(loader));

export { MSFT_sRGBFactors };
//# sourceMappingURL=MSFT_sRGBFactors-Hq6L5Rqj.js.map
