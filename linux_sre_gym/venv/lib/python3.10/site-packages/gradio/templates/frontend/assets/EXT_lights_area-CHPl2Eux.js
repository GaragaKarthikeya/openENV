import { d as Tools, aI as Light, T as Texture, L as Logger, V as Vector3, an as Node, _ as __decorate, s as serialize, R as RegisterClass, C as Color3, bh as TransformNode, ao as Quaternion, aG as unregisterGLTFExtension, aH as registerGLTFExtension } from './index-CxR0Yf_7.js';
import { R as RawTexture } from './rawTexture-Bto4WZO2.js';
import { ArrayItem, GLTFLoader } from './glTFLoader-BbG05oe0.js';
import './index-BvBk1Iap.js';
import './bone-C6xIrJX4.js';
import './skeleton-DzANFt7R.js';
import './assetContainer-7GPvH2S_.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * Loads LTC texture data from Babylon.js CDN.
 * @returns Promise with data for LTC1 and LTC2 textures for area lights.
 */
async function DecodeLTCTextureDataAsync() {
    const ltc1 = new Uint16Array(64 * 64 * 4);
    const ltc2 = new Uint16Array(64 * 64 * 4);
    const file = await Tools.LoadFileAsync(Tools.GetAssetUrl("https://assets.babylonjs.com/core/areaLights/areaLightsLTC.bin"));
    const ltcEncoded = new Uint16Array(file);
    const pixelCount = ltcEncoded.length / 8;
    for (let pixelIndex = 0; pixelIndex < pixelCount; pixelIndex++) {
        ltc1[pixelIndex * 4] = ltcEncoded[pixelIndex * 8];
        ltc1[pixelIndex * 4 + 1] = ltcEncoded[pixelIndex * 8 + 1];
        ltc1[pixelIndex * 4 + 2] = ltcEncoded[pixelIndex * 8 + 2];
        ltc1[pixelIndex * 4 + 3] = ltcEncoded[pixelIndex * 8 + 3];
        ltc2[pixelIndex * 4] = ltcEncoded[pixelIndex * 8 + 4];
        ltc2[pixelIndex * 4 + 1] = ltcEncoded[pixelIndex * 8 + 5];
        ltc2[pixelIndex * 4 + 2] = ltcEncoded[pixelIndex * 8 + 6];
        ltc2[pixelIndex * 4 + 3] = ltcEncoded[pixelIndex * 8 + 7];
    }
    return [ltc1, ltc2];
}

function CreateSceneLTCTextures(scene) {
    const useDelayedTextureLoading = scene.useDelayedTextureLoading;
    scene.useDelayedTextureLoading = false;
    const previousState = scene._blockEntityCollection;
    scene._blockEntityCollection = false;
    scene._ltcTextures = {
        LTC1: RawTexture.CreateRGBATexture(null, 64, 64, scene.getEngine(), false, false, 2, 2, 0, false, true),
        LTC2: RawTexture.CreateRGBATexture(null, 64, 64, scene.getEngine(), false, false, 2, 2, 0, false, true),
    };
    scene._blockEntityCollection = previousState;
    scene._ltcTextures.LTC1.wrapU = Texture.CLAMP_ADDRESSMODE;
    scene._ltcTextures.LTC1.wrapV = Texture.CLAMP_ADDRESSMODE;
    scene._ltcTextures.LTC2.wrapU = Texture.CLAMP_ADDRESSMODE;
    scene._ltcTextures.LTC2.wrapV = Texture.CLAMP_ADDRESSMODE;
    scene.useDelayedTextureLoading = useDelayedTextureLoading;
    DecodeLTCTextureDataAsync()
        // eslint-disable-next-line github/no-then
        .then((textureData) => {
        if (scene._ltcTextures) {
            const ltc1 = scene._ltcTextures?.LTC1;
            ltc1.update(textureData[0]);
            const ltc2 = scene._ltcTextures?.LTC2;
            ltc2.update(textureData[1]);
            scene.onDisposeObservable.addOnce(() => {
                scene._ltcTextures?.LTC1.dispose();
                scene._ltcTextures?.LTC2.dispose();
            });
        }
    })
        // eslint-disable-next-line github/no-then
        .catch((error) => {
        Logger.Error(`Area Light fail to get LTC textures data. Error: ${error}`);
    });
}
/**
 * Abstract Area Light class that servers as parent for all Area Lights implementations.
 * The light is emitted from the area in the -Z direction.
 */
class AreaLight extends Light {
    /**
     * Creates a area light object.
     * Documentation : https://doc.babylonjs.com/features/featuresDeepDive/lights/lights_introduction
     * @param name The friendly name of the light
     * @param position The position of the area light.
     * @param scene The scene the light belongs to
     * @param dontAddToScene True to not add the light to the scene
     */
    constructor(name, position, scene, dontAddToScene) {
        super(name, scene, dontAddToScene);
        this.position = position;
        if (!this._scene._ltcTextures) {
            CreateSceneLTCTextures(this._scene);
        }
    }
    transferTexturesToEffect(effect) {
        if (this._scene._ltcTextures) {
            effect.setTexture("areaLightsLTC1Sampler", this._scene._ltcTextures.LTC1);
            effect.setTexture("areaLightsLTC2Sampler", this._scene._ltcTextures.LTC2);
        }
        return this;
    }
    /**
     * Prepares the list of defines specific to the light type.
     * @param defines the list of defines
     * @param lightIndex defines the index of the light for the effect
     */
    prepareLightSpecificDefines(defines, lightIndex) {
        defines["AREALIGHT" + lightIndex] = true;
        defines["AREALIGHTUSED"] = true;
    }
    _isReady() {
        if (this._scene._ltcTextures) {
            return this._scene._ltcTextures.LTC1.isReady() && this._scene._ltcTextures.LTC2.isReady();
        }
        return false;
    }
}

Node.AddNodeConstructor("Light_Type_4", (name, scene) => {
    return () => new RectAreaLight(name, Vector3.Zero(), 1, 1, scene);
});
/**
 * A rectangular area light defined by an unique point in world space, a width and a height.
 * The light is emitted from the rectangular area in the -Z direction.
 */
class RectAreaLight extends AreaLight {
    /**
     * Rect Area Light width.
     */
    get width() {
        return this._width.x;
    }
    /**
     * Rect Area Light width.
     */
    set width(value) {
        this._width.x = value;
    }
    /**
     * Rect Area Light height.
     */
    get height() {
        return this._height.y;
    }
    /**
     * Rect Area Light height.
     */
    set height(value) {
        this._height.y = value;
    }
    /**
     * Creates a rectangular area light object.
     * Documentation : https://doc.babylonjs.com/features/featuresDeepDive/lights/lights_introduction
     * @param name The friendly name of the light
     * @param position The position of the area light.
     * @param width The width of the area light.
     * @param height The height of the area light.
     * @param scene The scene the light belongs to
     * @param dontAddToScene True to not add the light to the scene
     */
    constructor(name, position, width, height, scene, dontAddToScene) {
        super(name, position, scene, dontAddToScene);
        this._width = new Vector3(width, 0, 0);
        this._height = new Vector3(0, height, 0);
        this._pointTransformedPosition = Vector3.Zero();
        this._pointTransformedWidth = Vector3.Zero();
        this._pointTransformedHeight = Vector3.Zero();
    }
    /**
     * Returns the string "RectAreaLight"
     * @returns the class name
     */
    getClassName() {
        return "RectAreaLight";
    }
    /**
     * Returns the integer 4.
     * @returns The light Type id as a constant defines in Light.LIGHTTYPEID_x
     */
    // eslint-disable-next-line @typescript-eslint/naming-convention
    getTypeID() {
        return Light.LIGHTTYPEID_RECT_AREALIGHT;
    }
    _buildUniformLayout() {
        this._uniformBuffer.addUniform("vLightData", 4);
        this._uniformBuffer.addUniform("vLightDiffuse", 4);
        this._uniformBuffer.addUniform("vLightSpecular", 4);
        this._uniformBuffer.addUniform("vLightWidth", 4);
        this._uniformBuffer.addUniform("vLightHeight", 4);
        this._uniformBuffer.addUniform("shadowsInfo", 3);
        this._uniformBuffer.addUniform("depthValues", 2);
        this._uniformBuffer.create();
    }
    _computeTransformedInformation() {
        if (this.parent && this.parent.getWorldMatrix) {
            Vector3.TransformCoordinatesToRef(this.position, this.parent.getWorldMatrix(), this._pointTransformedPosition);
            Vector3.TransformNormalToRef(this._width, this.parent.getWorldMatrix(), this._pointTransformedWidth);
            Vector3.TransformNormalToRef(this._height, this.parent.getWorldMatrix(), this._pointTransformedHeight);
            return true;
        }
        return false;
    }
    /**
     * Sets the passed Effect "effect" with the PointLight transformed position (or position, if none) and passed name (string).
     * @param effect The effect to update
     * @param lightIndex The index of the light in the effect to update
     * @returns The point light
     */
    transferToEffect(effect, lightIndex) {
        const offset = this._scene.floatingOriginOffset;
        if (this._computeTransformedInformation()) {
            this._uniformBuffer.updateFloat4("vLightData", this._pointTransformedPosition.x - offset.x, this._pointTransformedPosition.y - offset.y, this._pointTransformedPosition.z - offset.z, 0, lightIndex);
            this._uniformBuffer.updateFloat4("vLightWidth", this._pointTransformedWidth.x / 2, this._pointTransformedWidth.y / 2, this._pointTransformedWidth.z / 2, 0, lightIndex);
            this._uniformBuffer.updateFloat4("vLightHeight", this._pointTransformedHeight.x / 2, this._pointTransformedHeight.y / 2, this._pointTransformedHeight.z / 2, 0, lightIndex);
        }
        else {
            this._uniformBuffer.updateFloat4("vLightData", this.position.x - offset.x, this.position.y - offset.y, this.position.z - offset.z, 0, lightIndex);
            this._uniformBuffer.updateFloat4("vLightWidth", this._width.x / 2, this._width.y / 2, this._width.z / 2, 0.0, lightIndex);
            this._uniformBuffer.updateFloat4("vLightHeight", this._height.x / 2, this._height.y / 2, this._height.z / 2, 0.0, lightIndex);
        }
        return this;
    }
    transferToNodeMaterialEffect(effect, lightDataUniformName) {
        const offset = this._scene.floatingOriginOffset;
        if (this._computeTransformedInformation()) {
            effect.setFloat3(lightDataUniformName, this._pointTransformedPosition.x - offset.x, this._pointTransformedPosition.y - offset.y, this._pointTransformedPosition.z - offset.z);
        }
        else {
            effect.setFloat3(lightDataUniformName, this.position.x - offset.x, this.position.y - offset.y, this.position.z - offset.z);
        }
        return this;
    }
}
__decorate([
    serialize()
], RectAreaLight.prototype, "width", null);
__decorate([
    serialize()
], RectAreaLight.prototype, "height", null);
// Register Class Name
RegisterClass("BABYLON.RectAreaLight", RectAreaLight);

const NAME = "EXT_lights_area";
/**
 * [Specification](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_lights_area/README.md)
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
class EXT_lights_area {
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
    loadNodeAsync(context, node, assign) {
        return GLTFLoader.LoadExtensionAsync(context, node, this.name, async (extensionContext, extension) => {
            this._loader._allMaterialsDirtyRequired = true;
            return await this._loader.loadNodeAsync(context, node, (babylonMesh) => {
                let babylonLight;
                const light = ArrayItem.Get(extensionContext, this._lights, extension.light);
                const name = light.name || babylonMesh.name;
                this._loader.babylonScene._blockEntityCollection = !!this._loader._assetContainer;
                const size = light.size !== undefined ? light.size : 1.0;
                switch (light.type) {
                    case "rect" /* EXTLightsArea_LightType.RECT */: {
                        const width = light.rect?.aspect !== undefined ? light.rect.aspect * size : size;
                        const height = size;
                        const babylonRectAreaLight = new RectAreaLight(name, Vector3.Zero(), width, height, this._loader.babylonScene);
                        babylonLight = babylonRectAreaLight;
                        break;
                    }
                    case "disk" /* EXTLightsArea_LightType.DISK */: {
                        // For disk lights, we'll use a rectangle light with the same area to approximate the disk light
                        // In the future, this could be extended to support actual disk area lights
                        const newSize = Math.sqrt(size * size * 0.25 * Math.PI); // Area of the disk
                        const babylonRectAreaLight = new RectAreaLight(name, Vector3.Zero(), newSize, newSize, this._loader.babylonScene);
                        babylonLight = babylonRectAreaLight;
                        break;
                    }
                    default: {
                        this._loader.babylonScene._blockEntityCollection = false;
                        throw new Error(`${extensionContext}: Invalid area light type (${light.type})`);
                    }
                }
                babylonLight._parentContainer = this._loader._assetContainer;
                this._loader.babylonScene._blockEntityCollection = false;
                light._babylonLight = babylonLight;
                babylonLight.falloffType = Light.FALLOFF_GLTF;
                babylonLight.diffuse = light.color ? Color3.FromArray(light.color) : Color3.White();
                babylonLight.intensity = light.intensity == undefined ? 1 : light.intensity;
                // glTF EXT_lights_area specifies lights face down -Z, but Babylon.js area lights face down +Z
                // Create a parent transform node with 180-degree rotation around Y axis to flip the direction
                const lightParentNode = new TransformNode(`${name}_orientation`, this._loader.babylonScene);
                lightParentNode.rotationQuaternion = Quaternion.RotationAxis(Vector3.Up(), Math.PI);
                lightParentNode.parent = babylonMesh;
                babylonLight.parent = lightParentNode;
                this._loader._babylonLights.push(babylonLight);
                GLTFLoader.AddPointerMetadata(babylonLight, extensionContext);
                assign(babylonMesh);
            });
        });
    }
}
unregisterGLTFExtension(NAME);
registerGLTFExtension(NAME, true, (loader) => new EXT_lights_area(loader));

export { EXT_lights_area };
//# sourceMappingURL=EXT_lights_area-CHPl2Eux.js.map
