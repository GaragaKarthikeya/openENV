import { a as SceneComponentConstants, b as Scene } from './index-CxR0Yf_7.js';
import { G as GeometryBufferRenderer } from './geometryBufferRenderer-Dvq35Qkr.js';
import './index-BvBk1Iap.js';
import './clipPlaneFragment-Vm6wRK67.js';
import './bumpFragment-Dxop61vv.js';
import './helperFunctions-CiFVmLcr.js';
import './bakedVertexAnimation-HQ8JUcrA.js';
import './morphTargetsVertex-C9IVzlEb.js';
import './instancesDeclaration-Bv0KzShp.js';
import './sceneUboDeclaration-CbezEhV0.js';
import './clipPlaneVertex-BKsOyoIZ.js';
import './bumpVertex-fS00LARh.js';

Object.defineProperty(Scene.prototype, "geometryBufferRenderer", {
    get: function () {
        return this._geometryBufferRenderer;
    },
    set: function (value) {
        if (value && value.isSupported) {
            this._geometryBufferRenderer = value;
        }
    },
    enumerable: true,
    configurable: true,
});
Scene.prototype.enableGeometryBufferRenderer = function (ratio = 1, depthFormat = 15, textureTypesAndFormats) {
    if (this._geometryBufferRenderer) {
        return this._geometryBufferRenderer;
    }
    this._geometryBufferRenderer = new GeometryBufferRenderer(this, ratio, depthFormat, textureTypesAndFormats);
    if (!this._geometryBufferRenderer.isSupported) {
        this._geometryBufferRenderer = null;
    }
    return this._geometryBufferRenderer;
};
Scene.prototype.disableGeometryBufferRenderer = function () {
    if (!this._geometryBufferRenderer) {
        return;
    }
    this._geometryBufferRenderer.dispose();
    this._geometryBufferRenderer = null;
};
/**
 * Defines the Geometry Buffer scene component responsible to manage a G-Buffer useful
 * in several rendering techniques.
 */
class GeometryBufferRendererSceneComponent {
    /**
     * Creates a new instance of the component for the given scene
     * @param scene Defines the scene to register the component in
     */
    constructor(scene) {
        /**
         * The component name helpful to identify the component in the list of scene components.
         */
        this.name = SceneComponentConstants.NAME_GEOMETRYBUFFERRENDERER;
        this.scene = scene;
    }
    /**
     * Registers the component in a given scene
     */
    register() {
        this.scene._gatherRenderTargetsStage.registerStep(SceneComponentConstants.STEP_GATHERRENDERTARGETS_GEOMETRYBUFFERRENDERER, this, this._gatherRenderTargets);
    }
    /**
     * Rebuilds the elements related to this component in case of
     * context lost for instance.
     */
    rebuild() {
        // Nothing to do for this component
    }
    /**
     * Disposes the component and the associated resources
     */
    dispose() {
        // Nothing to do for this component
    }
    _gatherRenderTargets(renderTargets) {
        if (this.scene._geometryBufferRenderer) {
            renderTargets.push(this.scene._geometryBufferRenderer.getGBuffer());
        }
    }
}
GeometryBufferRenderer._SceneComponentInitialization = (scene) => {
    // Register the G Buffer component to the scene.
    let component = scene._getComponent(SceneComponentConstants.NAME_GEOMETRYBUFFERRENDERER);
    if (!component) {
        component = new GeometryBufferRendererSceneComponent(scene);
        scene._addComponent(component);
    }
};

export { GeometryBufferRendererSceneComponent };
//# sourceMappingURL=geometryBufferRendererSceneComponent-890wIbQI.js.map
