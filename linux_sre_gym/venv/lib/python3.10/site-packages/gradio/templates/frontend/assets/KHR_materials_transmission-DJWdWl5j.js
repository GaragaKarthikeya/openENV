import { GLTFLoader } from './glTFLoader-BbG05oe0.js';
import { aK as Constants, O as Observable, d as Tools, aL as RenderTargetTexture, aG as unregisterGLTFExtension, aH as registerGLTFExtension } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';
import './bone-C6xIrJX4.js';
import './skeleton-DzANFt7R.js';
import './rawTexture-Bto4WZO2.js';
import './assetContainer-7GPvH2S_.js';
import './objectModelMapping-DnUo8NMz.js';

/**
 * A class to handle setting up the rendering of opaque objects to be shown through transmissive objects.
 */
class TransmissionHelper {
    /**
     * Creates the default options for the helper.
     * @returns the default options
     */
    static _GetDefaultOptions() {
        return {
            renderSize: 1024,
            samples: 4,
            lodGenerationScale: 1,
            lodGenerationOffset: -4,
            renderTargetTextureType: Constants.TEXTURETYPE_HALF_FLOAT,
            generateMipmaps: true,
        };
    }
    /**
     * constructor
     * @param options Defines the options we want to customize the helper
     * @param scene The scene to add the material to
     */
    constructor(options, scene) {
        this._opaqueRenderTarget = null;
        this._opaqueMeshesCache = [];
        this._transparentMeshesCache = [];
        this._materialObservers = {};
        this._options = {
            ...TransmissionHelper._GetDefaultOptions(),
            ...options,
        };
        this._scene = scene;
        this._scene._transmissionHelper = this;
        this.onErrorObservable = new Observable();
        this._scene.onDisposeObservable.addOnce(() => {
            this.dispose();
        });
        this._parseScene();
        this._setupRenderTargets();
    }
    /**
     * Updates the background according to the new options
     * @param options
     */
    updateOptions(options) {
        // First check if any options are actually being changed. If not, exit.
        const newValues = Object.keys(options).filter((key) => this._options[key] !== options[key]);
        if (!newValues.length) {
            return;
        }
        const newOptions = {
            ...this._options,
            ...options,
        };
        const oldOptions = this._options;
        this._options = newOptions;
        // If size changes, recreate everything
        if (newOptions.renderSize !== oldOptions.renderSize ||
            newOptions.renderTargetTextureType !== oldOptions.renderTargetTextureType ||
            newOptions.generateMipmaps !== oldOptions.generateMipmaps ||
            !this._opaqueRenderTarget) {
            this._setupRenderTargets();
        }
        else {
            this._opaqueRenderTarget.samples = newOptions.samples;
            this._opaqueRenderTarget.lodGenerationScale = newOptions.lodGenerationScale;
            this._opaqueRenderTarget.lodGenerationOffset = newOptions.lodGenerationOffset;
        }
    }
    /**
     * @returns the opaque render target texture or null if not available.
     */
    getOpaqueTarget() {
        return this._opaqueRenderTarget;
    }
    _shouldRenderAsTransmission(material) {
        return material?.subSurface?.isRefractionEnabled ? true : false;
    }
    _addMesh(mesh) {
        this._materialObservers[mesh.uniqueId] = mesh.onMaterialChangedObservable.add(this._onMeshMaterialChanged.bind(this));
        // we need to defer the processing because _addMesh may be called as part as an instance mesh creation, in which case some
        // internal properties are not setup yet, like _sourceMesh (needed when doing mesh.material below)
        Tools.SetImmediate(() => {
            if (this._shouldRenderAsTransmission(mesh.material)) {
                mesh.material.refractionTexture = this._opaqueRenderTarget;
                if (this._transparentMeshesCache.indexOf(mesh) === -1) {
                    this._transparentMeshesCache.push(mesh);
                }
            }
            else {
                if (this._opaqueMeshesCache.indexOf(mesh) === -1) {
                    this._opaqueMeshesCache.push(mesh);
                }
            }
        });
    }
    _removeMesh(mesh) {
        mesh.onMaterialChangedObservable.remove(this._materialObservers[mesh.uniqueId]);
        delete this._materialObservers[mesh.uniqueId];
        let idx = this._transparentMeshesCache.indexOf(mesh);
        if (idx !== -1) {
            this._transparentMeshesCache.splice(idx, 1);
        }
        idx = this._opaqueMeshesCache.indexOf(mesh);
        if (idx !== -1) {
            this._opaqueMeshesCache.splice(idx, 1);
        }
    }
    _parseScene() {
        this._scene.meshes.forEach(this._addMesh.bind(this));
        // Listen for when a mesh is added to the scene and add it to our cache lists.
        this._scene.onNewMeshAddedObservable.add(this._addMesh.bind(this));
        // Listen for when a mesh is removed from to the scene and remove it from our cache lists.
        this._scene.onMeshRemovedObservable.add(this._removeMesh.bind(this));
    }
    // When one of the meshes in the scene has its material changed, make sure that it's in the correct cache list.
    _onMeshMaterialChanged(mesh) {
        const transparentIdx = this._transparentMeshesCache.indexOf(mesh);
        const opaqueIdx = this._opaqueMeshesCache.indexOf(mesh);
        // If the material is transparent, make sure that it's added to the transparent list and removed from the opaque list
        const useTransmission = this._shouldRenderAsTransmission(mesh.material);
        if (useTransmission) {
            if (mesh.material) {
                const subSurface = mesh.material.subSurface;
                if (subSurface) {
                    subSurface.refractionTexture = this._opaqueRenderTarget;
                }
            }
            if (opaqueIdx !== -1) {
                this._opaqueMeshesCache.splice(opaqueIdx, 1);
                this._transparentMeshesCache.push(mesh);
            }
            else if (transparentIdx === -1) {
                this._transparentMeshesCache.push(mesh);
            }
            // If the material is opaque, make sure that it's added to the opaque list and removed from the transparent list
        }
        else {
            if (transparentIdx !== -1) {
                this._transparentMeshesCache.splice(transparentIdx, 1);
                this._opaqueMeshesCache.push(mesh);
            }
            else if (opaqueIdx === -1) {
                this._opaqueMeshesCache.push(mesh);
            }
        }
    }
    /**
     * @internal
     * Check if the opaque render target has not been disposed and can still be used.
     * @returns
     */
    _isRenderTargetValid() {
        return this._opaqueRenderTarget?.getInternalTexture() !== null;
    }
    /**
     * @internal
     * Setup the render targets according to the specified options.
     */
    _setupRenderTargets() {
        if (this._opaqueRenderTarget) {
            this._opaqueRenderTarget.dispose();
        }
        this._opaqueRenderTarget = new RenderTargetTexture("opaqueSceneTexture", this._options.renderSize, this._scene, this._options.generateMipmaps, undefined, this._options.renderTargetTextureType);
        this._opaqueRenderTarget.ignoreCameraViewport = true;
        this._opaqueRenderTarget.renderList = this._opaqueMeshesCache;
        this._opaqueRenderTarget.clearColor = this._options.clearColor?.clone() ?? this._scene.clearColor.clone();
        this._opaqueRenderTarget.gammaSpace = false;
        this._opaqueRenderTarget.lodGenerationScale = this._options.lodGenerationScale;
        this._opaqueRenderTarget.lodGenerationOffset = this._options.lodGenerationOffset;
        this._opaqueRenderTarget.samples = this._options.samples;
        this._opaqueRenderTarget.renderSprites = true;
        this._opaqueRenderTarget.renderParticles = true;
        this._opaqueRenderTarget.disableImageProcessing = true;
        let saveSceneEnvIntensity;
        this._opaqueRenderTarget.onBeforeBindObservable.add((opaqueRenderTarget) => {
            saveSceneEnvIntensity = this._scene.environmentIntensity;
            this._scene.environmentIntensity = 1.0;
            if (!this._options.clearColor) {
                this._scene.clearColor.toLinearSpaceToRef(opaqueRenderTarget.clearColor, this._scene.getEngine().useExactSrgbConversions);
            }
            else {
                opaqueRenderTarget.clearColor.copyFrom(this._options.clearColor);
            }
        });
        this._opaqueRenderTarget.onAfterUnbindObservable.add(() => {
            this._scene.environmentIntensity = saveSceneEnvIntensity;
        });
        for (const mesh of this._transparentMeshesCache) {
            if (this._shouldRenderAsTransmission(mesh.material)) {
                mesh.material.refractionTexture = this._opaqueRenderTarget;
            }
        }
    }
    /**
     * Dispose all the elements created by the Helper.
     */
    dispose() {
        this._scene._transmissionHelper = undefined;
        if (this._opaqueRenderTarget) {
            this._opaqueRenderTarget.dispose();
            this._opaqueRenderTarget = null;
        }
        this._transparentMeshesCache = [];
        this._opaqueMeshesCache = [];
    }
}
const NAME = "KHR_materials_transmission";
/**
 * [Specification](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_transmission/README.md)
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
class KHR_materials_transmission {
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
        this.order = 175;
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
            promises.push(this._loadTransparentPropertiesAsync(extensionContext, material, babylonMaterial, extension));
            // eslint-disable-next-line github/no-then
            return await Promise.all(promises).then(() => { });
        });
    }
    // eslint-disable-next-line no-restricted-syntax, @typescript-eslint/promise-function-async
    _loadTransparentPropertiesAsync(context, material, babylonMaterial, extension) {
        const adapter = this._loader._getOrCreateMaterialAdapter(babylonMaterial);
        const transmissionWeight = extension.transmissionFactor !== undefined ? extension.transmissionFactor : 0.0;
        if (transmissionWeight === 0) {
            return Promise.resolve();
        }
        // Set transmission properties immediately via adapter
        adapter.configureTransmission();
        adapter.transmissionWeight = transmissionWeight;
        // Handle transmission helper setup (only needed for PBR materials)
        if (transmissionWeight > 0 && !this._loader.parent.dontUseTransmissionHelper) {
            const scene = babylonMaterial.getScene();
            if (!scene._transmissionHelper) {
                new TransmissionHelper({}, babylonMaterial.getScene());
            }
            else if (!scene._transmissionHelper?._isRenderTargetValid()) {
                // If the render target is not valid, recreate it.
                scene._transmissionHelper?._setupRenderTargets();
            }
        }
        // Load texture if present
        let texturePromise = Promise.resolve(null);
        if (extension.transmissionTexture) {
            extension.transmissionTexture.nonColorData = true;
            texturePromise = this._loader.loadTextureInfoAsync(`${context}/transmissionTexture`, extension.transmissionTexture, (texture) => {
                texture.name = `${babylonMaterial.name} (Transmission)`;
                adapter.transmissionWeightTexture = texture;
            });
        }
        // eslint-disable-next-line github/no-then
        return texturePromise.then(() => { });
    }
}
unregisterGLTFExtension(NAME);
registerGLTFExtension(NAME, true, (loader) => new KHR_materials_transmission(loader));

export { KHR_materials_transmission };
//# sourceMappingURL=KHR_materials_transmission-DJWdWl5j.js.map
