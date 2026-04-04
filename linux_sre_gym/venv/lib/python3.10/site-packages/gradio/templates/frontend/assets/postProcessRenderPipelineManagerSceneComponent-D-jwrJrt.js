import { O as Observable, a as SceneComponentConstants, b as Scene } from './index-CxR0Yf_7.js';
import './index-BvBk1Iap.js';

/**
 * PostProcessRenderPipelineManager class
 * @see https://doc.babylonjs.com/features/featuresDeepDive/postProcesses/postProcessRenderPipeline
 */
class PostProcessRenderPipelineManager {
    /**
     * Initializes a PostProcessRenderPipelineManager
     * @see https://doc.babylonjs.com/features/featuresDeepDive/postProcesses/postProcessRenderPipeline
     */
    constructor() {
        this._renderPipelines = {};
        this._onNewPipelineAddedObservable = new Observable();
        this._onPipelineRemovedObservable = new Observable();
    }
    /**
     * An event triggered when a pipeline is added to the manager
     */
    get onNewPipelineAddedObservable() {
        return this._onNewPipelineAddedObservable;
    }
    /**
     * An event triggered when a pipeline is removed from the manager
     */
    get onPipelineRemovedObservable() {
        return this._onPipelineRemovedObservable;
    }
    /**
     * Gets the list of supported render pipelines
     */
    get supportedPipelines() {
        const result = [];
        for (const renderPipelineName in this._renderPipelines) {
            if (Object.prototype.hasOwnProperty.call(this._renderPipelines, renderPipelineName)) {
                const pipeline = this._renderPipelines[renderPipelineName];
                if (pipeline.isSupported) {
                    result.push(pipeline);
                }
            }
        }
        return result;
    }
    /**
     * Adds a pipeline to the manager
     * @param renderPipeline The pipeline to add
     */
    addPipeline(renderPipeline) {
        this.removePipeline(renderPipeline._name);
        this._renderPipelines[renderPipeline._name] = renderPipeline;
        this._onNewPipelineAddedObservable.notifyObservers(renderPipeline);
    }
    /**
     * Remove the pipeline from the manager
     * @param renderPipelineName the name of the pipeline to remove
     */
    removePipeline(renderPipelineName) {
        const pipeline = this._renderPipelines[renderPipelineName];
        if (pipeline) {
            this._onPipelineRemovedObservable.notifyObservers(pipeline);
            delete this._renderPipelines[renderPipelineName];
        }
    }
    /**
     * Attaches a camera to the pipeline
     * @param renderPipelineName The name of the pipeline to attach to
     * @param cameras the camera to attach
     * @param unique if the camera can be attached multiple times to the pipeline
     */
    attachCamerasToRenderPipeline(renderPipelineName, cameras, unique = false) {
        const renderPipeline = this._renderPipelines[renderPipelineName];
        if (!renderPipeline) {
            return;
        }
        renderPipeline._attachCameras(cameras, unique);
    }
    /**
     * Detaches a camera from the pipeline
     * @param renderPipelineName The name of the pipeline to detach from
     * @param cameras the camera to detach
     */
    detachCamerasFromRenderPipeline(renderPipelineName, cameras) {
        const renderPipeline = this._renderPipelines[renderPipelineName];
        if (!renderPipeline) {
            return;
        }
        renderPipeline._detachCameras(cameras);
    }
    /**
     * Enables an effect by name on a pipeline
     * @param renderPipelineName the name of the pipeline to enable the effect in
     * @param renderEffectName the name of the effect to enable
     * @param cameras the cameras that the effect should be enabled on
     */
    enableEffectInPipeline(renderPipelineName, renderEffectName, cameras) {
        const renderPipeline = this._renderPipelines[renderPipelineName];
        if (!renderPipeline) {
            return;
        }
        renderPipeline._enableEffect(renderEffectName, cameras);
    }
    /**
     * Disables an effect by name on a pipeline
     * @param renderPipelineName the name of the pipeline to disable the effect in
     * @param renderEffectName the name of the effect to disable
     * @param cameras the cameras that the effect should be disabled on
     */
    disableEffectInPipeline(renderPipelineName, renderEffectName, cameras) {
        const renderPipeline = this._renderPipelines[renderPipelineName];
        if (!renderPipeline) {
            return;
        }
        renderPipeline._disableEffect(renderEffectName, cameras);
    }
    /**
     * Updates the state of all contained render pipelines and disposes of any non supported pipelines
     */
    update() {
        for (const renderPipelineName in this._renderPipelines) {
            if (Object.prototype.hasOwnProperty.call(this._renderPipelines, renderPipelineName)) {
                const pipeline = this._renderPipelines[renderPipelineName];
                if (!pipeline.isSupported) {
                    pipeline.dispose();
                    delete this._renderPipelines[renderPipelineName];
                }
                else {
                    pipeline._update();
                }
            }
        }
    }
    /** @internal */
    _rebuild() {
        for (const renderPipelineName in this._renderPipelines) {
            if (Object.prototype.hasOwnProperty.call(this._renderPipelines, renderPipelineName)) {
                const pipeline = this._renderPipelines[renderPipelineName];
                pipeline._rebuild();
            }
        }
    }
    /**
     * Disposes of the manager and pipelines
     */
    dispose() {
        for (const renderPipelineName in this._renderPipelines) {
            if (Object.prototype.hasOwnProperty.call(this._renderPipelines, renderPipelineName)) {
                const pipeline = this._renderPipelines[renderPipelineName];
                pipeline.dispose();
            }
        }
    }
}

Object.defineProperty(Scene.prototype, "postProcessRenderPipelineManager", {
    get: function () {
        if (!this._postProcessRenderPipelineManager) {
            // Register the G Buffer component to the scene.
            let component = this._getComponent(SceneComponentConstants.NAME_POSTPROCESSRENDERPIPELINEMANAGER);
            if (!component) {
                component = new PostProcessRenderPipelineManagerSceneComponent(this);
                this._addComponent(component);
            }
            this._postProcessRenderPipelineManager = new PostProcessRenderPipelineManager();
        }
        return this._postProcessRenderPipelineManager;
    },
    enumerable: true,
    configurable: true,
});
/**
 * Defines the Render Pipeline scene component responsible to rendering pipelines
 */
class PostProcessRenderPipelineManagerSceneComponent {
    /**
     * Creates a new instance of the component for the given scene
     * @param scene Defines the scene to register the component in
     */
    constructor(scene) {
        /**
         * The component name helpful to identify the component in the list of scene components.
         */
        this.name = SceneComponentConstants.NAME_POSTPROCESSRENDERPIPELINEMANAGER;
        this.scene = scene;
    }
    /**
     * Registers the component in a given scene
     */
    register() {
        this.scene._gatherRenderTargetsStage.registerStep(SceneComponentConstants.STEP_GATHERRENDERTARGETS_POSTPROCESSRENDERPIPELINEMANAGER, this, this._gatherRenderTargets);
    }
    /**
     * Rebuilds the elements related to this component in case of
     * context lost for instance.
     */
    rebuild() {
        if (this.scene._postProcessRenderPipelineManager) {
            this.scene._postProcessRenderPipelineManager._rebuild();
        }
    }
    /**
     * Disposes the component and the associated resources
     */
    dispose() {
        if (this.scene._postProcessRenderPipelineManager) {
            this.scene._postProcessRenderPipelineManager.dispose();
        }
    }
    _gatherRenderTargets() {
        if (this.scene._postProcessRenderPipelineManager) {
            this.scene._postProcessRenderPipelineManager.update();
        }
    }
}

export { PostProcessRenderPipelineManagerSceneComponent };
//# sourceMappingURL=postProcessRenderPipelineManagerSceneComponent-D-jwrJrt.js.map
