import { V as Vector3, aI as Light, C as Color3, T as Texture, aG as unregisterGLTFExtension, aH as registerGLTFExtension } from './index-CxR0Yf_7.js';
import { a as SpotLight } from './objectModelMapping-DnUo8NMz.js';
import { ArrayItem, GLTFLoader } from './glTFLoader-BbG05oe0.js';
import './index-BvBk1Iap.js';
import './bone-C6xIrJX4.js';
import './skeleton-DzANFt7R.js';
import './rawTexture-Bto4WZO2.js';
import './assetContainer-7GPvH2S_.js';

const NAME = "EXT_lights_ies";
/**
 * [Specification](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Vendor/EXT_lights_ies)
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
class EXT_lights_ies {
    /**
     * @internal
     */
    constructor(loader) {
        /**
         * The name of this extension.
         */
        this.name = NAME;
        this._loader = loader;
        this.enabled = this._loader.isExtensionUsed(NAME);
    }
    /** @internal */
    dispose() {
        this._loader = null;
        delete this._lights;
    }
    /** @internal */
    onLoading() {
        const extensions = this._loader.gltf.extensions;
        if (extensions && extensions[this.name]) {
            const extension = extensions[this.name];
            this._lights = extension.lights;
            ArrayItem.Assign(this._lights);
        }
    }
    /**
     * @internal
     */
    // eslint-disable-next-line no-restricted-syntax
    // eslint-disable-next-line no-restricted-syntax
    loadNodeAsync(context, node, assign) {
        return GLTFLoader.LoadExtensionAsync(context, node, this.name, async (extensionContext, extension) => {
            this._loader._allMaterialsDirtyRequired = true;
            let babylonSpotLight;
            let light;
            const transformNode = await this._loader.loadNodeAsync(context, node, (babylonMesh) => {
                light = ArrayItem.Get(extensionContext, this._lights, extension.light);
                const name = light.name || babylonMesh.name;
                this._loader.babylonScene._blockEntityCollection = !!this._loader._assetContainer;
                babylonSpotLight = new SpotLight(name, Vector3.Zero(), Vector3.Backward(), 0, 1, this._loader.babylonScene);
                babylonSpotLight.angle = Math.PI / 2;
                babylonSpotLight.innerAngle = 0;
                babylonSpotLight._parentContainer = this._loader._assetContainer;
                this._loader.babylonScene._blockEntityCollection = false;
                light._babylonLight = babylonSpotLight;
                babylonSpotLight.falloffType = Light.FALLOFF_GLTF;
                babylonSpotLight.diffuse = extension.color ? Color3.FromArray(extension.color) : Color3.White();
                babylonSpotLight.intensity = extension.multiplier || 1;
                babylonSpotLight.range = Number.MAX_VALUE;
                babylonSpotLight.parent = babylonMesh;
                this._loader._babylonLights.push(babylonSpotLight);
                GLTFLoader.AddPointerMetadata(babylonSpotLight, extensionContext);
                assign(babylonMesh);
            });
            // Load the profile
            let bufferData;
            if (light.uri) {
                bufferData = await this._loader.loadUriAsync(context, light, light.uri);
            }
            else {
                const bufferView = ArrayItem.Get(`${context}/bufferView`, this._loader.gltf.bufferViews, light.bufferView);
                bufferData = await this._loader.loadBufferViewAsync(`/bufferViews/${bufferView.index}`, bufferView);
            }
            babylonSpotLight.iesProfileTexture = new Texture(name + "_iesProfile", this._loader.babylonScene, true, false, undefined, null, null, bufferData, true, undefined, undefined, undefined, undefined, ".ies");
            return transformNode;
        });
    }
}
unregisterGLTFExtension(NAME);
registerGLTFExtension(NAME, true, (loader) => new EXT_lights_ies(loader));

export { EXT_lights_ies };
//# sourceMappingURL=EXT_lights_ies-CU7g-1Zx.js.map
