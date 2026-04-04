import { bk as ShadowLight, aI as Light, V as Vector3, M as Matrix, an as Node, _ as __decorate, s as serialize, R as RegisterClass, bl as DirectionalLight, C as Color3, aG as unregisterGLTFExtension, aH as registerGLTFExtension } from './index-CxR0Yf_7.js';
import { a as SpotLight } from './objectModelMapping-DnUo8NMz.js';
import { ArrayItem, GLTFLoader } from './glTFLoader-BbG05oe0.js';
import './index-BvBk1Iap.js';
import './bone-C6xIrJX4.js';
import './skeleton-DzANFt7R.js';
import './rawTexture-Bto4WZO2.js';
import './assetContainer-7GPvH2S_.js';

Node.AddNodeConstructor("Light_Type_0", (name, scene) => {
    return () => new PointLight(name, Vector3.Zero(), scene);
});
/**
 * A point light is a light defined by an unique point in world space.
 * The light is emitted in every direction from this point.
 * A good example of a point light is a standard light bulb.
 * Documentation: https://doc.babylonjs.com/features/featuresDeepDive/lights/lights_introduction
 */
class PointLight extends ShadowLight {
    /**
     * Getter: In case of direction provided, the shadow will not use a cube texture but simulate a spot shadow as a fallback
     * This specifies what angle the shadow will use to be created.
     *
     * It default to 90 degrees to work nicely with the cube texture generation for point lights shadow maps.
     */
    get shadowAngle() {
        return this._shadowAngle;
    }
    /**
     * Setter: In case of direction provided, the shadow will not use a cube texture but simulate a spot shadow as a fallback
     * This specifies what angle the shadow will use to be created.
     *
     * It default to 90 degrees to work nicely with the cube texture generation for point lights shadow maps.
     */
    set shadowAngle(value) {
        this._shadowAngle = value;
        this.forceProjectionMatrixCompute();
    }
    /**
     * Gets the direction if it has been set.
     * In case of direction provided, the shadow will not use a cube texture but simulate a spot shadow as a fallback
     */
    get direction() {
        return this._direction;
    }
    /**
     * In case of direction provided, the shadow will not use a cube texture but simulate a spot shadow as a fallback
     */
    set direction(value) {
        const previousNeedCube = this.needCube();
        this._direction = value;
        if (this.needCube() !== previousNeedCube && this._shadowGenerators) {
            const iterator = this._shadowGenerators.values();
            for (let key = iterator.next(); key.done !== true; key = iterator.next()) {
                const shadowGenerator = key.value;
                shadowGenerator.recreateShadowMap();
            }
        }
    }
    /**
     * Creates a PointLight object from the passed name and position (Vector3) and adds it in the scene.
     * A PointLight emits the light in every direction.
     * It can cast shadows.
     * If the scene camera is already defined and you want to set your PointLight at the camera position, just set it :
     * ```javascript
     * var pointLight = new PointLight("pl", camera.position, scene);
     * ```
     * Documentation : https://doc.babylonjs.com/features/featuresDeepDive/lights/lights_introduction
     * @param name The light friendly name
     * @param position The position of the point light in the scene
     * @param scene The scene the lights belongs to
     * @param dontAddToScene True to not add the light to the scene
     */
    constructor(name, position, scene, dontAddToScene) {
        super(name, scene, dontAddToScene);
        this._shadowAngle = Math.PI / 2;
        this.position = position;
    }
    /**
     * Returns the string "PointLight"
     * @returns the class name
     */
    getClassName() {
        return "PointLight";
    }
    /**
     * Returns the integer 0.
     * @returns The light Type id as a constant defines in Light.LIGHTTYPEID_x
     */
    // eslint-disable-next-line @typescript-eslint/naming-convention
    getTypeID() {
        return Light.LIGHTTYPEID_POINTLIGHT;
    }
    /**
     * Specifies whether or not the shadowmap should be a cube texture.
     * @returns true if the shadowmap needs to be a cube texture.
     */
    needCube() {
        return !this.direction;
    }
    /**
     * Returns a new Vector3 aligned with the PointLight cube system according to the passed cube face index (integer).
     * @param faceIndex The index of the face we are computed the direction to generate shadow
     * @returns The set direction in 2d mode otherwise the direction to the cubemap face if needCube() is true
     */
    getShadowDirection(faceIndex) {
        if (this.direction) {
            return super.getShadowDirection(faceIndex);
        }
        else {
            switch (faceIndex) {
                case 0:
                    return new Vector3(1.0, 0.0, 0.0);
                case 1:
                    return new Vector3(-1, 0.0, 0.0);
                case 2:
                    return new Vector3(0.0, -1, 0.0);
                case 3:
                    return new Vector3(0.0, 1.0, 0.0);
                case 4:
                    return new Vector3(0.0, 0.0, 1.0);
                case 5:
                    return new Vector3(0.0, 0.0, -1);
            }
        }
        return Vector3.Zero();
    }
    /**
     * Sets the passed matrix "matrix" as a left-handed perspective projection matrix with the following settings :
     * - fov = PI / 2
     * - aspect ratio : 1.0
     * - z-near and far equal to the active camera minZ and maxZ.
     * Returns the PointLight.
     * @param matrix
     * @param viewMatrix
     * @param renderList
     */
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _setDefaultShadowProjectionMatrix(matrix, viewMatrix, renderList) {
        const activeCamera = this.getScene().activeCamera;
        const minZ = this.getDepthMinZ(activeCamera);
        const maxZ = this.getDepthMaxZ(activeCamera);
        const useReverseDepthBuffer = this.getScene().getEngine().useReverseDepthBuffer;
        Matrix.PerspectiveFovLHToRef(this.shadowAngle, 1.0, useReverseDepthBuffer ? maxZ : minZ, useReverseDepthBuffer ? minZ : maxZ, matrix, true, this._scene.getEngine().isNDCHalfZRange, undefined, useReverseDepthBuffer);
    }
    _buildUniformLayout() {
        this._uniformBuffer.addUniform("vLightData", 4);
        this._uniformBuffer.addUniform("vLightDiffuse", 4);
        this._uniformBuffer.addUniform("vLightSpecular", 4);
        this._uniformBuffer.addUniform("vLightFalloff", 4);
        this._uniformBuffer.addUniform("shadowsInfo", 3);
        this._uniformBuffer.addUniform("depthValues", 2);
        this._uniformBuffer.create();
    }
    /**
     * Sets the passed Effect "effect" with the PointLight transformed position (or position, if none) and passed name (string).
     * @param effect The effect to update
     * @param lightIndex The index of the light in the effect to update
     * @returns The point light
     */
    transferToEffect(effect, lightIndex) {
        const offset = this._scene.floatingOriginOffset;
        if (this.computeTransformedInformation()) {
            this._uniformBuffer.updateFloat4("vLightData", this.transformedPosition.x - offset.x, this.transformedPosition.y - offset.y, this.transformedPosition.z - offset.z, 0.0, lightIndex);
        }
        else {
            this._uniformBuffer.updateFloat4("vLightData", this.position.x - offset.x, this.position.y - offset.y, this.position.z - offset.z, 0, lightIndex);
        }
        this._uniformBuffer.updateFloat4("vLightFalloff", this.range, this._inverseSquaredRange, 0, 0, lightIndex);
        return this;
    }
    transferToNodeMaterialEffect(effect, lightDataUniformName) {
        const offset = this._scene.floatingOriginOffset;
        if (this.computeTransformedInformation()) {
            effect.setFloat3(lightDataUniformName, this.transformedPosition.x - offset.x, this.transformedPosition.y - offset.y, this.transformedPosition.z - offset.z);
        }
        else {
            effect.setFloat3(lightDataUniformName, this.position.x - offset.x, this.position.y - offset.y, this.position.z - offset.z);
        }
        return this;
    }
    /**
     * Prepares the list of defines specific to the light type.
     * @param defines the list of defines
     * @param lightIndex defines the index of the light for the effect
     */
    prepareLightSpecificDefines(defines, lightIndex) {
        defines["POINTLIGHT" + lightIndex] = true;
    }
}
__decorate([
    serialize()
], PointLight.prototype, "shadowAngle", null);
// Register Class Name
RegisterClass("BABYLON.PointLight", PointLight);

const NAME = "KHR_lights_punctual";
/**
 * [Specification](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md)
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
class KHR_lights {
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
                switch (light.type) {
                    case "directional" /* KHRLightsPunctual_LightType.DIRECTIONAL */: {
                        const babylonDirectionalLight = new DirectionalLight(name, Vector3.Backward(), this._loader.babylonScene);
                        babylonDirectionalLight.position.setAll(0);
                        babylonLight = babylonDirectionalLight;
                        break;
                    }
                    case "point" /* KHRLightsPunctual_LightType.POINT */: {
                        babylonLight = new PointLight(name, Vector3.Zero(), this._loader.babylonScene);
                        break;
                    }
                    case "spot" /* KHRLightsPunctual_LightType.SPOT */: {
                        const babylonSpotLight = new SpotLight(name, Vector3.Zero(), Vector3.Backward(), 0, 1, this._loader.babylonScene);
                        babylonSpotLight.angle = ((light.spot && light.spot.outerConeAngle) || Math.PI / 4) * 2;
                        babylonSpotLight.innerAngle = ((light.spot && light.spot.innerConeAngle) || 0) * 2;
                        babylonLight = babylonSpotLight;
                        break;
                    }
                    default: {
                        this._loader.babylonScene._blockEntityCollection = false;
                        throw new Error(`${extensionContext}: Invalid light type (${light.type})`);
                    }
                }
                babylonLight._parentContainer = this._loader._assetContainer;
                this._loader.babylonScene._blockEntityCollection = false;
                light._babylonLight = babylonLight;
                babylonLight.falloffType = Light.FALLOFF_GLTF;
                babylonLight.diffuse = light.color ? Color3.FromArray(light.color) : Color3.White();
                babylonLight.intensity = light.intensity == undefined ? 1 : light.intensity;
                babylonLight.range = light.range == undefined ? Number.MAX_VALUE : light.range;
                babylonLight.parent = babylonMesh;
                this._loader._babylonLights.push(babylonLight);
                GLTFLoader.AddPointerMetadata(babylonLight, extensionContext);
                assign(babylonMesh);
            });
        });
    }
}
unregisterGLTFExtension(NAME);
registerGLTFExtension(NAME, true, (loader) => new KHR_lights(loader));

export { KHR_lights };
//# sourceMappingURL=KHR_lights_punctual-D5iAnnKo.js.map
