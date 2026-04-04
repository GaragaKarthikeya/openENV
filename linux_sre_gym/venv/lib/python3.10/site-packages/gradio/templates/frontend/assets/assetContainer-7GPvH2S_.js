import { bg as AbstractMesh, d as Tools, L as Logger, J as VertexBuffer, bh as TransformNode, M as Matrix, c as TmpVectors, at as DeepCopier, l as Mesh, R as RegisterClass, aq as EngineStore, aI as Light, b1 as Camera, bi as Tags } from './index-CxR0Yf_7.js';

Mesh._instancedMeshFactory = (name, mesh) => {
    const instance = new InstancedMesh(name, mesh);
    if (mesh.instancedBuffers) {
        instance.instancedBuffers = {};
        for (const key in mesh.instancedBuffers) {
            instance.instancedBuffers[key] = mesh.instancedBuffers[key];
        }
    }
    return instance;
};
/**
 * Creates an instance based on a source mesh.
 */
class InstancedMesh extends AbstractMesh {
    /**
     * Creates a new InstancedMesh object from the mesh source.
     * @param name defines the name of the instance
     * @param source the mesh to create the instance from
     */
    constructor(name, source) {
        super(name, source.getScene());
        /** @internal */
        this._indexInSourceMeshInstanceArray = -1;
        /** @internal */
        this._distanceToCamera = 0;
        source.addInstance(this);
        this._sourceMesh = source;
        this._unIndexed = source._unIndexed;
        this.position.copyFrom(source.position);
        this.rotation.copyFrom(source.rotation);
        this.scaling.copyFrom(source.scaling);
        if (source.rotationQuaternion) {
            this.rotationQuaternion = source.rotationQuaternion.clone();
        }
        this.animations = source.animations.slice();
        for (const range of source.getAnimationRanges()) {
            if (range != null) {
                this.createAnimationRange(range.name, range.from, range.to);
            }
        }
        this.infiniteDistance = source.infiniteDistance;
        this.setPivotMatrix(source.getPivotMatrix());
        if (!source.skeleton && !source.morphTargetManager && source.hasBoundingInfo) {
            // without skeleton or morphTargetManager, use bounding info of source mesh directly
            const boundingInfo = source.getBoundingInfo();
            this.buildBoundingInfo(boundingInfo.minimum, boundingInfo.maximum);
        }
        else {
            this.refreshBoundingInfo(true, true);
        }
        this._syncSubMeshes();
    }
    /**
     * @returns the string "InstancedMesh".
     */
    getClassName() {
        return "InstancedMesh";
    }
    /** Gets the list of lights affecting that mesh */
    get lightSources() {
        return this._sourceMesh._lightSources;
    }
    _resyncLightSources() {
        // Do nothing as all the work will be done by source mesh
    }
    _resyncLightSource() {
        // Do nothing as all the work will be done by source mesh
    }
    _removeLightSource() {
        // Do nothing as all the work will be done by source mesh
    }
    // Methods
    /**
     * If the source mesh receives shadows
     */
    get receiveShadows() {
        return this._sourceMesh.receiveShadows;
    }
    set receiveShadows(_value) {
        if (this._sourceMesh?.receiveShadows !== _value) {
            Tools.Warn("Setting receiveShadows on an instanced mesh has no effect");
        }
    }
    /**
     * The material of the source mesh
     */
    get material() {
        return this._sourceMesh.material;
    }
    set material(_value) {
        if (this._sourceMesh?.material !== _value) {
            Tools.Warn("Setting material on an instanced mesh has no effect");
        }
    }
    /**
     * Visibility of the source mesh
     */
    get visibility() {
        return this._sourceMesh.visibility;
    }
    set visibility(_value) {
        if (this._sourceMesh?.visibility !== _value) {
            Tools.Warn("Setting visibility on an instanced mesh has no effect");
        }
    }
    /**
     * Skeleton of the source mesh
     */
    get skeleton() {
        return this._sourceMesh.skeleton;
    }
    set skeleton(_value) {
        if (this._sourceMesh?.skeleton !== _value) {
            Tools.Warn("Setting skeleton on an instanced mesh has no effect");
        }
    }
    /**
     * Rendering ground id of the source mesh
     */
    get renderingGroupId() {
        return this._sourceMesh.renderingGroupId;
    }
    set renderingGroupId(value) {
        if (!this._sourceMesh || value === this._sourceMesh.renderingGroupId) {
            return;
        }
        //no-op with warning
        Logger.Warn("Note - setting renderingGroupId of an instanced mesh has no effect on the scene");
    }
    /**
     * @returns the total number of vertices (integer).
     */
    getTotalVertices() {
        return this._sourceMesh ? this._sourceMesh.getTotalVertices() : 0;
    }
    /**
     * Returns a positive integer : the total number of indices in this mesh geometry.
     * @returns the number of indices or zero if the mesh has no geometry.
     */
    getTotalIndices() {
        return this._sourceMesh.getTotalIndices();
    }
    /**
     * The source mesh of the instance
     */
    get sourceMesh() {
        return this._sourceMesh;
    }
    /**
     * Gets the mesh internal Geometry object
     */
    get geometry() {
        return this._sourceMesh._geometry;
    }
    /**
     * Creates a new InstancedMesh object from the mesh model.
     * @see https://doc.babylonjs.com/features/featuresDeepDive/mesh/copies/instances
     * @param name defines the name of the new instance
     * @returns a new InstancedMesh
     */
    createInstance(name) {
        return this._sourceMesh.createInstance(name);
    }
    /**
     * Is this node ready to be used/rendered
     * @param completeCheck defines if a complete check (including materials and lights) has to be done (false by default)
     * @returns {boolean} is it ready
     */
    isReady(completeCheck = false) {
        return this._sourceMesh.isReady(completeCheck, true);
    }
    /**
     * Returns an array of integers or a typed array (Int32Array, Uint32Array, Uint16Array) populated with the mesh indices.
     * @param kind kind of verticies to retrieve (eg. positions, normals, uvs, etc.)
     * @param copyWhenShared If true (default false) and and if the mesh geometry is shared among some other meshes, the returned array is a copy of the internal one.
     * @param forceCopy defines a boolean forcing the copy of the buffer no matter what the value of copyWhenShared is
     * @returns a float array or a Float32Array of the requested kind of data : positions, normals, uvs, etc.
     */
    getVerticesData(kind, copyWhenShared, forceCopy) {
        return this._sourceMesh.getVerticesData(kind, copyWhenShared, forceCopy);
    }
    copyVerticesData(kind, vertexData) {
        this._sourceMesh.copyVerticesData(kind, vertexData);
    }
    getVertexBuffer(kind, bypassInstanceData) {
        return this._sourceMesh.getVertexBuffer(kind, bypassInstanceData);
    }
    /**
     * Sets the vertex data of the mesh geometry for the requested `kind`.
     * If the mesh has no geometry, a new Geometry object is set to the mesh and then passed this vertex data.
     * The `data` are either a numeric array either a Float32Array.
     * The parameter `updatable` is passed as is to the underlying Geometry object constructor (if initially none) or updater.
     * The parameter `stride` is an optional positive integer, it is usually automatically deducted from the `kind` (3 for positions or normals, 2 for UV, etc).
     * Note that a new underlying VertexBuffer object is created each call.
     * If the `kind` is the `PositionKind`, the mesh BoundingInfo is renewed, so the bounding box and sphere, and the mesh World Matrix is recomputed.
     *
     * Possible `kind` values :
     * - VertexBuffer.PositionKind
     * - VertexBuffer.UVKind
     * - VertexBuffer.UV2Kind
     * - VertexBuffer.UV3Kind
     * - VertexBuffer.UV4Kind
     * - VertexBuffer.UV5Kind
     * - VertexBuffer.UV6Kind
     * - VertexBuffer.ColorKind
     * - VertexBuffer.MatricesIndicesKind
     * - VertexBuffer.MatricesIndicesExtraKind
     * - VertexBuffer.MatricesWeightsKind
     * - VertexBuffer.MatricesWeightsExtraKind
     *
     * Returns the Mesh.
     * @param kind defines vertex data kind
     * @param data defines the data source
     * @param updatable defines if the data must be flagged as updatable (false as default)
     * @param stride defines the vertex stride (optional)
     * @returns the current mesh
     */
    setVerticesData(kind, data, updatable, stride) {
        if (this.sourceMesh) {
            this.sourceMesh.setVerticesData(kind, data, updatable, stride);
        }
        return this.sourceMesh;
    }
    /**
     * Updates the existing vertex data of the mesh geometry for the requested `kind`.
     * If the mesh has no geometry, it is simply returned as it is.
     * The `data` are either a numeric array either a Float32Array.
     * No new underlying VertexBuffer object is created.
     * If the `kind` is the `PositionKind` and if `updateExtends` is true, the mesh BoundingInfo is renewed, so the bounding box and sphere, and the mesh World Matrix is recomputed.
     * If the parameter `makeItUnique` is true, a new global geometry is created from this positions and is set to the mesh.
     *
     * Possible `kind` values :
     * - VertexBuffer.PositionKind
     * - VertexBuffer.UVKind
     * - VertexBuffer.UV2Kind
     * - VertexBuffer.UV3Kind
     * - VertexBuffer.UV4Kind
     * - VertexBuffer.UV5Kind
     * - VertexBuffer.UV6Kind
     * - VertexBuffer.ColorKind
     * - VertexBuffer.MatricesIndicesKind
     * - VertexBuffer.MatricesIndicesExtraKind
     * - VertexBuffer.MatricesWeightsKind
     * - VertexBuffer.MatricesWeightsExtraKind
     *
     * Returns the Mesh.
     * @param kind defines vertex data kind
     * @param data defines the data source
     * @param updateExtends defines if extends info of the mesh must be updated (can be null). This is mostly useful for "position" kind
     * @param makeItUnique defines it the updated vertex buffer must be flagged as unique (false by default)
     * @returns the source mesh
     */
    updateVerticesData(kind, data, updateExtends, makeItUnique) {
        if (this.sourceMesh) {
            this.sourceMesh.updateVerticesData(kind, data, updateExtends, makeItUnique);
        }
        return this.sourceMesh;
    }
    /**
     * Sets the mesh indices.
     * Expects an array populated with integers or a typed array (Int32Array, Uint32Array, Uint16Array).
     * If the mesh has no geometry, a new Geometry object is created and set to the mesh.
     * This method creates a new index buffer each call.
     * Returns the Mesh.
     * @param indices the source data
     * @param totalVertices defines the total number of vertices referenced by indices (could be null)
     * @returns source mesh
     */
    setIndices(indices, totalVertices = null) {
        if (this.sourceMesh) {
            this.sourceMesh.setIndices(indices, totalVertices);
        }
        return this.sourceMesh;
    }
    /**
     * Boolean : True if the mesh owns the requested kind of data.
     * @param kind defines which buffer to check (positions, indices, normals, etc). Possible `kind` values :
     * - VertexBuffer.PositionKind
     * - VertexBuffer.UVKind
     * - VertexBuffer.UV2Kind
     * - VertexBuffer.UV3Kind
     * - VertexBuffer.UV4Kind
     * - VertexBuffer.UV5Kind
     * - VertexBuffer.UV6Kind
     * - VertexBuffer.ColorKind
     * - VertexBuffer.MatricesIndicesKind
     * - VertexBuffer.MatricesIndicesExtraKind
     * - VertexBuffer.MatricesWeightsKind
     * - VertexBuffer.MatricesWeightsExtraKind
     * @returns true if data kind is present
     */
    isVerticesDataPresent(kind) {
        return this._sourceMesh.isVerticesDataPresent(kind);
    }
    /**
     * @returns an array of indices (IndicesArray).
     */
    getIndices() {
        return this._sourceMesh.getIndices();
    }
    get _positions() {
        return this._sourceMesh._positions;
    }
    refreshBoundingInfo(applySkeletonOrOptions = false, applyMorph = false) {
        if (this.hasBoundingInfo && this.getBoundingInfo().isLocked) {
            return this;
        }
        let options;
        if (typeof applySkeletonOrOptions === "object") {
            options = applySkeletonOrOptions;
        }
        else {
            options = {
                applySkeleton: applySkeletonOrOptions,
                applyMorph: applyMorph,
            };
        }
        const bias = this._sourceMesh.geometry ? this._sourceMesh.geometry.boundingBias : null;
        this._refreshBoundingInfo(this._sourceMesh._getData(options, null, VertexBuffer.PositionKind), bias);
        return this;
    }
    /** @internal */
    _preActivate() {
        if (this._currentLOD) {
            this._currentLOD._preActivate();
        }
        return this;
    }
    /**
     * @internal
     */
    _activate(renderId, intermediateRendering) {
        super._activate(renderId, intermediateRendering);
        if (!this._sourceMesh.subMeshes) {
            Logger.Warn("Instances should only be created for meshes with geometry.");
        }
        if (this._currentLOD) {
            const differentSign = this._currentLOD._getWorldMatrixDeterminant() >= 0 !== this._getWorldMatrixDeterminant() >= 0;
            if (differentSign) {
                this._internalAbstractMeshDataInfo._actAsRegularMesh = true;
                return true;
            }
            this._internalAbstractMeshDataInfo._actAsRegularMesh = false;
            this._currentLOD._registerInstanceForRenderId(this, renderId);
            if (intermediateRendering) {
                if (!this._currentLOD._internalAbstractMeshDataInfo._isActiveIntermediate) {
                    this._currentLOD._internalAbstractMeshDataInfo._onlyForInstancesIntermediate = true;
                    return true;
                }
            }
            else {
                if (!this._currentLOD._internalAbstractMeshDataInfo._isActive) {
                    this._currentLOD._internalAbstractMeshDataInfo._onlyForInstances = true;
                    return true;
                }
            }
        }
        return false;
    }
    /** @internal */
    _postActivate() {
        if (this._sourceMesh.edgesShareWithInstances && this._sourceMesh._edgesRenderer && this._sourceMesh._edgesRenderer.isEnabled && this._sourceMesh._renderingGroup) {
            // we are using the edge renderer of the source mesh
            this._sourceMesh._renderingGroup._edgesRenderers.pushNoDuplicate(this._sourceMesh._edgesRenderer);
            this._sourceMesh._edgesRenderer.customInstances.push(this.getWorldMatrix());
        }
        else if (this._edgesRenderer && this._edgesRenderer.isEnabled && this._sourceMesh._renderingGroup) {
            // we are using the edge renderer defined for this instance
            this._sourceMesh._renderingGroup._edgesRenderers.push(this._edgesRenderer);
        }
    }
    getWorldMatrix() {
        if (this._currentLOD &&
            this._currentLOD !== this._sourceMesh &&
            this._currentLOD.billboardMode !== TransformNode.BILLBOARDMODE_NONE &&
            this._currentLOD._masterMesh !== this) {
            if (!this._billboardWorldMatrix) {
                this._billboardWorldMatrix = new Matrix();
            }
            const tempMaster = this._currentLOD._masterMesh;
            this._currentLOD._masterMesh = this;
            TmpVectors.Vector3[7].copyFrom(this._currentLOD.position);
            this._currentLOD.position.set(0, 0, 0);
            this._billboardWorldMatrix.copyFrom(this._currentLOD.computeWorldMatrix(true));
            this._currentLOD.position.copyFrom(TmpVectors.Vector3[7]);
            this._currentLOD._masterMesh = tempMaster;
            return this._billboardWorldMatrix;
        }
        return super.getWorldMatrix();
    }
    get isAnInstance() {
        return true;
    }
    /**
     * Returns the current associated LOD AbstractMesh.
     * @param camera defines the camera to use to pick the LOD level
     * @returns a Mesh or `null` if no LOD is associated with the AbstractMesh
     */
    getLOD(camera) {
        if (!camera) {
            return this;
        }
        const sourceMeshLODLevels = this.sourceMesh.getLODLevels();
        if (!sourceMeshLODLevels || sourceMeshLODLevels.length === 0) {
            this._currentLOD = this.sourceMesh;
        }
        else {
            const boundingInfo = this.getBoundingInfo();
            this._currentLOD = this.sourceMesh.getLOD(camera, boundingInfo.boundingSphere);
        }
        return this._currentLOD;
    }
    /**
     * @internal
     */
    _preActivateForIntermediateRendering(renderId) {
        return this.sourceMesh._preActivateForIntermediateRendering(renderId);
    }
    /** @internal */
    _syncSubMeshes() {
        this.releaseSubMeshes();
        if (this._sourceMesh.subMeshes) {
            for (let index = 0; index < this._sourceMesh.subMeshes.length; index++) {
                this._sourceMesh.subMeshes[index].clone(this, this._sourceMesh);
            }
        }
        return this;
    }
    /** @internal */
    _generatePointsArray() {
        return this._sourceMesh._generatePointsArray();
    }
    /** @internal */
    _updateBoundingInfo() {
        if (this.hasBoundingInfo) {
            this.getBoundingInfo().update(this.worldMatrixFromCache);
        }
        else {
            this.buildBoundingInfo(this.absolutePosition, this.absolutePosition, this.worldMatrixFromCache);
        }
        this._updateSubMeshesBoundingInfo(this.worldMatrixFromCache);
        return this;
    }
    /**
     * Creates a new InstancedMesh from the current mesh.
     *
     * Returns the clone.
     * @param name the cloned mesh name
     * @param newParent the optional Node to parent the clone to.
     * @param doNotCloneChildren if `true` the model children aren't cloned.
     * @param newSourceMesh if set this mesh will be used as the source mesh instead of ths instance's one
     * @returns the clone
     */
    clone(name, newParent = null, doNotCloneChildren, newSourceMesh) {
        const result = (newSourceMesh || this._sourceMesh).createInstance(name);
        // Deep copy
        DeepCopier.DeepCopy(this, result, [
            "name",
            "subMeshes",
            "uniqueId",
            "parent",
            "lightSources",
            "receiveShadows",
            "material",
            "visibility",
            "skeleton",
            "sourceMesh",
            "isAnInstance",
            "facetNb",
            "isFacetDataEnabled",
            "isBlocked",
            "useBones",
            "hasInstances",
            "collider",
            "edgesRenderer",
            "forward",
            "up",
            "right",
            "absolutePosition",
            "absoluteScaling",
            "absoluteRotationQuaternion",
            "isWorldMatrixFrozen",
            "nonUniformScaling",
            "behaviors",
            "worldMatrixFromCache",
            "hasThinInstances",
            "hasBoundingInfo",
            "geometry",
        ], []);
        // Parent
        if (newParent) {
            result.parent = newParent;
        }
        if (!doNotCloneChildren) {
            // Children
            for (let index = 0; index < this.getScene().meshes.length; index++) {
                const mesh = this.getScene().meshes[index];
                if (mesh.parent === this) {
                    mesh.clone(mesh.name, result);
                }
            }
        }
        result.computeWorldMatrix(true);
        this.onClonedObservable.notifyObservers(result);
        return result;
    }
    /**
     * Disposes the InstancedMesh.
     * Returns nothing.
     * @param doNotRecurse Set to true to not recurse into each children (recurse into each children by default)
     * @param disposeMaterialAndTextures Set to true to also dispose referenced materials and textures (false by default)
     */
    dispose(doNotRecurse, disposeMaterialAndTextures = false) {
        // Remove from mesh
        this._sourceMesh.removeInstance(this);
        super.dispose(doNotRecurse, disposeMaterialAndTextures);
    }
    /**
     * @internal
     */
    _serializeAsParent(serializationObject) {
        super._serializeAsParent(serializationObject);
        serializationObject.parentId = this._sourceMesh.uniqueId;
        serializationObject.parentInstanceIndex = this._indexInSourceMeshInstanceArray;
    }
    /**
     * Instantiate (when possible) or clone that node with its hierarchy
     * @param newParent defines the new parent to use for the instance (or clone)
     * @param options defines options to configure how copy is done
     * @param options.doNotInstantiate defines if the model must be instantiated or just cloned
     * @param options.newSourcedMesh newSourcedMesh the new source mesh for the instance (or clone)
     * @param onNewNodeCreated defines an option callback to call when a clone or an instance is created
     * @returns an instance (or a clone) of the current node with its hierarchy
     */
    instantiateHierarchy(newParent = null, options, onNewNodeCreated) {
        const clone = this.clone("Clone of " + (this.name || this.id), newParent || this.parent, true, options && options.newSourcedMesh);
        if (clone) {
            if (onNewNodeCreated) {
                onNewNodeCreated(this, clone);
            }
        }
        for (const child of this.getChildTransformNodes(true)) {
            child.instantiateHierarchy(clone, options, onNewNodeCreated);
        }
        return clone;
    }
}
Mesh.prototype.registerInstancedBuffer = function (kind, stride) {
    // Remove existing one
    this._userInstancedBuffersStorage?.vertexBuffers[kind]?.dispose();
    // Creates the instancedBuffer field if not present
    if (!this.instancedBuffers) {
        this.instancedBuffers = {};
        for (const instance of this.instances) {
            instance.instancedBuffers = {};
        }
    }
    if (!this._userInstancedBuffersStorage) {
        this._userInstancedBuffersStorage = {
            data: {},
            vertexBuffers: {},
            strides: {},
            sizes: {},
            vertexArrayObjects: this.getEngine().getCaps().vertexArrayObject ? {} : undefined,
        };
    }
    // Creates an empty property for this kind
    this.instancedBuffers[kind] = null;
    this._userInstancedBuffersStorage.strides[kind] = stride;
    this._userInstancedBuffersStorage.sizes[kind] = stride * 32; // Initial size
    this._userInstancedBuffersStorage.data[kind] = new Float32Array(this._userInstancedBuffersStorage.sizes[kind]);
    this._userInstancedBuffersStorage.vertexBuffers[kind] = new VertexBuffer(this.getEngine(), this._userInstancedBuffersStorage.data[kind], kind, true, false, stride, true);
    for (const instance of this.instances) {
        instance.instancedBuffers[kind] = null;
    }
    this._invalidateInstanceVertexArrayObject();
    this._markSubMeshesAsAttributesDirty();
};
Mesh.prototype._processInstancedBuffers = function (visibleInstances, renderSelf) {
    const instanceCount = visibleInstances ? visibleInstances.length : 0;
    for (const kind in this.instancedBuffers) {
        let size = this._userInstancedBuffersStorage.sizes[kind];
        const stride = this._userInstancedBuffersStorage.strides[kind];
        // Resize if required
        const expectedSize = (instanceCount + 1) * stride;
        while (size < expectedSize) {
            size *= 2;
        }
        if (this._userInstancedBuffersStorage.data[kind].length != size) {
            this._userInstancedBuffersStorage.data[kind] = new Float32Array(size);
            this._userInstancedBuffersStorage.sizes[kind] = size;
            if (this._userInstancedBuffersStorage.vertexBuffers[kind]) {
                this._userInstancedBuffersStorage.vertexBuffers[kind].dispose();
                this._userInstancedBuffersStorage.vertexBuffers[kind] = null;
            }
        }
        const data = this._userInstancedBuffersStorage.data[kind];
        // Update data buffer
        let offset = 0;
        if (renderSelf) {
            const value = this.instancedBuffers[kind];
            if (value.toArray) {
                value.toArray(data, offset);
            }
            else if (value.copyToArray) {
                value.copyToArray(data, offset);
            }
            else {
                data[offset] = value;
            }
            offset += stride;
        }
        for (let instanceIndex = 0; instanceIndex < instanceCount; instanceIndex++) {
            const instance = visibleInstances[instanceIndex];
            const value = instance.instancedBuffers[kind];
            if (value.toArray) {
                value.toArray(data, offset);
            }
            else if (value.copyToArray) {
                value.copyToArray(data, offset);
            }
            else {
                data[offset] = value;
            }
            offset += stride;
        }
        // Update vertex buffer
        if (!this._userInstancedBuffersStorage.vertexBuffers[kind]) {
            this._userInstancedBuffersStorage.vertexBuffers[kind] = new VertexBuffer(this.getEngine(), this._userInstancedBuffersStorage.data[kind], kind, true, false, stride, true);
            this._invalidateInstanceVertexArrayObject();
        }
        else {
            this._userInstancedBuffersStorage.vertexBuffers[kind].updateDirectly(data, 0);
        }
    }
};
Mesh.prototype._invalidateInstanceVertexArrayObject = function () {
    if (!this._userInstancedBuffersStorage || this._userInstancedBuffersStorage.vertexArrayObjects === undefined) {
        return;
    }
    for (const kind in this._userInstancedBuffersStorage.vertexArrayObjects) {
        this.getEngine().releaseVertexArrayObject(this._userInstancedBuffersStorage.vertexArrayObjects[kind]);
    }
    this._userInstancedBuffersStorage.vertexArrayObjects = {};
};
Mesh.prototype._disposeInstanceSpecificData = function () {
    for (const renderPassId in this._instanceDataStorage.renderPasses) {
        this._instanceDataStorage.renderPasses[renderPassId].instancesBuffer?.dispose();
    }
    this._instanceDataStorage.renderPasses = {};
    while (this.instances.length) {
        this.instances[0].dispose();
    }
    for (const kind in this.instancedBuffers) {
        if (this._userInstancedBuffersStorage.vertexBuffers[kind]) {
            this._userInstancedBuffersStorage.vertexBuffers[kind].dispose();
        }
    }
    this._invalidateInstanceVertexArrayObject();
    this.instancedBuffers = {};
};
// Register Class Name
RegisterClass("BABYLON.InstancedMesh", InstancedMesh);

/**
 * Root class for AssetContainer and KeepAssets
 */
class AbstractAssetContainer {
    constructor() {
        /**
         * Gets the list of root nodes (ie. nodes with no parent)
         */
        this.rootNodes = [];
        /** All of the cameras added to this scene
         * @see https://doc.babylonjs.com/features/featuresDeepDive/cameras
         */
        this.cameras = [];
        /**
         * All of the lights added to this scene
         * @see https://doc.babylonjs.com/features/featuresDeepDive/lights/lights_introduction
         */
        this.lights = [];
        /**
         * All of the (abstract) meshes added to this scene
         */
        this.meshes = [];
        /**
         * The list of skeletons added to the scene
         * @see https://doc.babylonjs.com/features/featuresDeepDive/mesh/bonesSkeletons
         */
        this.skeletons = [];
        /**
         * All of the particle systems added to this scene
         * @see https://doc.babylonjs.com/features/featuresDeepDive/particles/particle_system/particle_system_intro
         */
        this.particleSystems = [];
        /**
         * Gets a list of Animations associated with the scene
         */
        this.animations = [];
        /**
         * All of the animation groups added to this scene
         * @see https://doc.babylonjs.com/features/featuresDeepDive/animation/groupAnimations
         */
        this.animationGroups = [];
        /**
         * All of the multi-materials added to this scene
         * @see https://doc.babylonjs.com/features/featuresDeepDive/materials/using/multiMaterials
         */
        this.multiMaterials = [];
        /**
         * All of the materials added to this scene
         * In the context of a Scene, it is not supposed to be modified manually.
         * Any addition or removal should be done using the addMaterial and removeMaterial Scene methods.
         * Note also that the order of the Material within the array is not significant and might change.
         * @see https://doc.babylonjs.com/features/featuresDeepDive/materials/using/materials_introduction
         */
        this.materials = [];
        /**
         * The list of morph target managers added to the scene
         * @see https://doc.babylonjs.com/features/featuresDeepDive/mesh/dynamicMeshMorph
         */
        this.morphTargetManagers = [];
        /**
         * The list of geometries used in the scene.
         */
        this.geometries = [];
        /**
         * All of the transform nodes added to this scene
         * In the context of a Scene, it is not supposed to be modified manually.
         * Any addition or removal should be done using the addTransformNode and removeTransformNode Scene methods.
         * Note also that the order of the TransformNode within the array is not significant and might change.
         * @see https://doc.babylonjs.com/features/featuresDeepDive/mesh/transforms/parent_pivot/transform_node
         */
        this.transformNodes = [];
        /**
         * ActionManagers available on the scene.
         * @deprecated
         */
        this.actionManagers = [];
        /**
         * Textures to keep.
         */
        this.textures = [];
        /** @internal */
        this._environmentTexture = null;
        /**
         * The list of postprocesses added to the scene
         */
        this.postProcesses = [];
        /**
         * The list of sounds
         */
        this.sounds = null;
        /**
         * The list of effect layers added to the scene
         */
        this.effectLayers = [];
        /**
         * The list of layers added to the scene
         */
        this.layers = [];
        /**
         * The list of reflection probes added to the scene
         */
        this.reflectionProbes = [];
    }
    /**
     * Texture used in all pbr material as the reflection texture.
     * As in the majority of the scene they are the same (exception for multi room and so on),
     * this is easier to reference from here than from all the materials.
     */
    get environmentTexture() {
        return this._environmentTexture;
    }
    set environmentTexture(value) {
        this._environmentTexture = value;
    }
    /**
     * @returns all meshes, lights, cameras, transformNodes and bones
     */
    getNodes() {
        let nodes = [];
        nodes = nodes.concat(this.meshes);
        nodes = nodes.concat(this.lights);
        nodes = nodes.concat(this.cameras);
        nodes = nodes.concat(this.transformNodes); // dummies
        for (const skeleton of this.skeletons) {
            nodes = nodes.concat(skeleton.bones);
        }
        return nodes;
    }
}
/**
 * Set of assets to keep when moving a scene into an asset container.
 */
class KeepAssets extends AbstractAssetContainer {
}
/**
 * Class used to store the output of the AssetContainer.instantiateAllMeshesToScene function
 */
class InstantiatedEntries {
    constructor() {
        /**
         * List of new root nodes (eg. nodes with no parent)
         */
        this.rootNodes = [];
        /**
         * List of new skeletons
         */
        this.skeletons = [];
        /**
         * List of new animation groups
         */
        this.animationGroups = [];
    }
    /**
     * Disposes the instantiated entries from the scene
     */
    dispose() {
        const rootNodes = this.rootNodes;
        for (const rootNode of rootNodes) {
            rootNode.dispose();
        }
        rootNodes.length = 0;
        const skeletons = this.skeletons;
        for (const skeleton of skeletons) {
            skeleton.dispose();
        }
        skeletons.length = 0;
        const animationGroups = this.animationGroups;
        for (const animationGroup of animationGroups) {
            animationGroup.dispose();
        }
        animationGroups.length = 0;
    }
}
/**
 * Container with a set of assets that can be added or removed from a scene.
 */
class AssetContainer extends AbstractAssetContainer {
    /**
     * Instantiates an AssetContainer.
     * @param scene The scene the AssetContainer belongs to.
     */
    constructor(scene) {
        super();
        this._wasAddedToScene = false;
        scene = scene || EngineStore.LastCreatedScene;
        if (!scene) {
            return;
        }
        this.scene = scene;
        this["proceduralTextures"] = [];
        scene.onDisposeObservable.add(() => {
            if (!this._wasAddedToScene) {
                this.dispose();
            }
        });
        this._onContextRestoredObserver = scene.getEngine().onContextRestoredObservable.add(() => {
            for (const geometry of this.geometries) {
                geometry._rebuild();
            }
            for (const mesh of this.meshes) {
                mesh._rebuild();
            }
            for (const system of this.particleSystems) {
                system.rebuild();
            }
            for (const texture of this.textures) {
                texture._rebuild();
            }
        });
    }
    /**
     * Given a list of nodes, return a topological sorting of them.
     * @param nodes
     * @returns a sorted array of nodes
     */
    _topologicalSort(nodes) {
        const nodesUidMap = new Map();
        for (const node of nodes) {
            nodesUidMap.set(node.uniqueId, node);
        }
        const dependencyGraph = {
            dependsOn: new Map(), // given a node id, what are the ids of the nodes it depends on
            dependedBy: new Map(), // given a node id, what are the ids of the nodes that depend on it
        };
        // Build the dependency graph given the list of nodes
        // First pass: Initialize the empty dependency graph
        for (const node of nodes) {
            const nodeId = node.uniqueId;
            dependencyGraph.dependsOn.set(nodeId, new Set());
            dependencyGraph.dependedBy.set(nodeId, new Set());
        }
        // Second pass: Populate the dependency graph. We assume that we
        // don't need to check for cycles here, as the scene graph cannot
        // contain cycles. Our graph also already contains all transitive
        // dependencies because getDescendants returns the transitive
        // dependencies by default.
        for (const node of nodes) {
            const nodeId = node.uniqueId;
            const dependsOn = dependencyGraph.dependsOn.get(nodeId);
            if (node instanceof InstancedMesh) {
                const masterMesh = node.sourceMesh;
                if (nodesUidMap.has(masterMesh.uniqueId)) {
                    dependsOn.add(masterMesh.uniqueId);
                    dependencyGraph.dependedBy.get(masterMesh.uniqueId).add(nodeId);
                }
            }
            const dependedBy = dependencyGraph.dependedBy.get(nodeId);
            for (const child of node.getDescendants()) {
                const childId = child.uniqueId;
                if (nodesUidMap.has(childId)) {
                    dependedBy.add(childId);
                    const childDependsOn = dependencyGraph.dependsOn.get(childId);
                    childDependsOn.add(nodeId);
                }
            }
        }
        // Third pass: Topological sort
        const sortedNodes = [];
        // First: Find all nodes that have no dependencies
        const leaves = [];
        for (const node of nodes) {
            const nodeId = node.uniqueId;
            if (dependencyGraph.dependsOn.get(nodeId).size === 0) {
                leaves.push(node);
                nodesUidMap.delete(nodeId);
            }
        }
        const visitList = leaves;
        while (visitList.length > 0) {
            const nodeToVisit = visitList.shift();
            sortedNodes.push(nodeToVisit);
            // Remove the node from the dependency graph
            // When a node is visited, we know that dependsOn is empty.
            // So we only need to remove the node from dependedBy.
            const dependedByVisitedNode = dependencyGraph.dependedBy.get(nodeToVisit.uniqueId);
            // Array.from(x.values()) is to make the TS compiler happy
            for (const dependedByVisitedNodeId of Array.from(dependedByVisitedNode.values())) {
                const dependsOnDependedByVisitedNode = dependencyGraph.dependsOn.get(dependedByVisitedNodeId);
                dependsOnDependedByVisitedNode.delete(nodeToVisit.uniqueId);
                if (dependsOnDependedByVisitedNode.size === 0 && nodesUidMap.get(dependedByVisitedNodeId)) {
                    visitList.push(nodesUidMap.get(dependedByVisitedNodeId));
                    nodesUidMap.delete(dependedByVisitedNodeId);
                }
            }
        }
        if (nodesUidMap.size > 0) {
            Logger.Error("SceneSerializer._topologicalSort: There were unvisited nodes:");
            nodesUidMap.forEach((node) => {
                Logger.Error(node.name);
            });
        }
        return sortedNodes;
    }
    _addNodeAndDescendantsToList(list, addedIds, rootNode, predicate) {
        if (!rootNode || (predicate && !predicate(rootNode)) || addedIds.has(rootNode.uniqueId)) {
            return;
        }
        list.push(rootNode);
        addedIds.add(rootNode.uniqueId);
        for (const child of rootNode.getDescendants(true)) {
            this._addNodeAndDescendantsToList(list, addedIds, child, predicate);
        }
    }
    /**
     * Check if a specific node is contained in this asset container.
     * @param node the node to check
     * @returns true if the node is contained in this container, otherwise false.
     */
    _isNodeInContainer(node) {
        if (node instanceof AbstractMesh && this.meshes.indexOf(node) !== -1) {
            return true;
        }
        if (node instanceof TransformNode && this.transformNodes.indexOf(node) !== -1) {
            return true;
        }
        if (node instanceof Light && this.lights.indexOf(node) !== -1) {
            return true;
        }
        if (node instanceof Camera && this.cameras.indexOf(node) !== -1) {
            return true;
        }
        return false;
    }
    /**
     * For every node in the scene, check if its parent node is also in the scene.
     * @returns true if every node's parent is also in the scene, otherwise false.
     */
    _isValidHierarchy() {
        for (const node of this.meshes) {
            if (node.parent && !this._isNodeInContainer(node.parent)) {
                Logger.Warn(`Node ${node.name} has a parent that is not in the container.`);
                return false;
            }
        }
        for (const node of this.transformNodes) {
            if (node.parent && !this._isNodeInContainer(node.parent)) {
                Logger.Warn(`Node ${node.name} has a parent that is not in the container.`);
                return false;
            }
        }
        for (const node of this.lights) {
            if (node.parent && !this._isNodeInContainer(node.parent)) {
                Logger.Warn(`Node ${node.name} has a parent that is not in the container.`);
                return false;
            }
        }
        for (const node of this.cameras) {
            if (node.parent && !this._isNodeInContainer(node.parent)) {
                Logger.Warn(`Node ${node.name} has a parent that is not in the container.`);
                return false;
            }
        }
        return true;
    }
    /**
     * Instantiate or clone all meshes and add the new ones to the scene.
     * Skeletons and animation groups will all be cloned
     * @param nameFunction defines an optional function used to get new names for clones
     * @param cloneMaterials defines an optional boolean that defines if materials must be cloned as well (false by default)
     * @param options defines an optional list of options to control how to instantiate / clone models
     * @param options.doNotInstantiate defines if the model must be instantiated or just cloned
     * @param options.predicate defines a predicate used to filter whih mesh to instantiate/clone
     * @returns a list of rootNodes, skeletons and animation groups that were duplicated
     */
    instantiateModelsToScene(nameFunction, cloneMaterials = false, options) {
        if (!this._isValidHierarchy()) {
            Tools.Warn("SceneSerializer.InstantiateModelsToScene: The Asset Container hierarchy is not valid.");
        }
        const conversionMap = {};
        const storeMap = {};
        const result = new InstantiatedEntries();
        const alreadySwappedSkeletons = [];
        const alreadySwappedMaterials = [];
        const localOptions = {
            doNotInstantiate: true,
            ...options,
        };
        const onClone = (source, clone) => {
            conversionMap[source.uniqueId] = clone.uniqueId;
            storeMap[clone.uniqueId] = clone;
            if (nameFunction) {
                clone.name = nameFunction(source.name);
            }
            if (clone instanceof Mesh) {
                const clonedMesh = clone;
                if (clonedMesh.morphTargetManager) {
                    const oldMorphTargetManager = source.morphTargetManager;
                    clonedMesh.morphTargetManager = oldMorphTargetManager.clone();
                    for (let index = 0; index < oldMorphTargetManager.numTargets; index++) {
                        const oldTarget = oldMorphTargetManager.getTarget(index);
                        const newTarget = clonedMesh.morphTargetManager.getTarget(index);
                        conversionMap[oldTarget.uniqueId] = newTarget.uniqueId;
                        storeMap[newTarget.uniqueId] = newTarget;
                    }
                }
            }
        };
        const nodesToSort = [];
        const idsOnSortList = new Set();
        for (const transformNode of this.transformNodes) {
            if (transformNode.parent === null) {
                this._addNodeAndDescendantsToList(nodesToSort, idsOnSortList, transformNode, localOptions.predicate);
            }
        }
        for (const mesh of this.meshes) {
            if (mesh.parent === null) {
                this._addNodeAndDescendantsToList(nodesToSort, idsOnSortList, mesh, localOptions.predicate);
            }
        }
        // Topologically sort nodes by parenting/instancing relationships so that all resources are in place
        // when a given node is instantiated.
        const sortedNodes = this._topologicalSort(nodesToSort);
        const onNewCreated = (source, clone) => {
            onClone(source, clone);
            if (source.parent) {
                const replicatedParentId = conversionMap[source.parent.uniqueId];
                const replicatedParent = storeMap[replicatedParentId];
                if (replicatedParent) {
                    clone.parent = replicatedParent;
                }
                else {
                    clone.parent = source.parent;
                }
            }
            if (clone.position && source.position) {
                clone.position.copyFrom(source.position);
            }
            if (clone.rotationQuaternion && source.rotationQuaternion) {
                clone.rotationQuaternion.copyFrom(source.rotationQuaternion);
            }
            if (clone.rotation && source.rotation) {
                clone.rotation.copyFrom(source.rotation);
            }
            if (clone.scaling && source.scaling) {
                clone.scaling.copyFrom(source.scaling);
            }
            if (clone.material) {
                const mesh = clone;
                if (mesh.material) {
                    if (cloneMaterials) {
                        const sourceMaterial = source.material;
                        if (alreadySwappedMaterials.indexOf(sourceMaterial) === -1) {
                            let swap = sourceMaterial.clone(nameFunction ? nameFunction(sourceMaterial.name) : "Clone of " + sourceMaterial.name);
                            alreadySwappedMaterials.push(sourceMaterial);
                            conversionMap[sourceMaterial.uniqueId] = swap.uniqueId;
                            storeMap[swap.uniqueId] = swap;
                            if (sourceMaterial.getClassName() === "MultiMaterial") {
                                const multi = sourceMaterial;
                                for (const material of multi.subMaterials) {
                                    if (!material) {
                                        continue;
                                    }
                                    swap = material.clone(nameFunction ? nameFunction(material.name) : "Clone of " + material.name);
                                    alreadySwappedMaterials.push(material);
                                    conversionMap[material.uniqueId] = swap.uniqueId;
                                    storeMap[swap.uniqueId] = swap;
                                }
                                multi.subMaterials = multi.subMaterials.map((m) => m && storeMap[conversionMap[m.uniqueId]]);
                            }
                        }
                        if (mesh.getClassName() !== "InstancedMesh") {
                            mesh.material = storeMap[conversionMap[sourceMaterial.uniqueId]];
                        }
                    }
                    else {
                        if (mesh.material.getClassName() === "MultiMaterial") {
                            if (this.scene.multiMaterials.indexOf(mesh.material) === -1) {
                                this.scene.addMultiMaterial(mesh.material);
                            }
                        }
                        else {
                            if (this.scene.materials.indexOf(mesh.material) === -1) {
                                this.scene.addMaterial(mesh.material);
                            }
                        }
                    }
                }
            }
            if (clone.parent === null) {
                result.rootNodes.push(clone);
            }
        };
        for (const node of sortedNodes) {
            if (node.getClassName() === "InstancedMesh") {
                const instancedNode = node;
                const sourceMesh = instancedNode.sourceMesh;
                const replicatedSourceId = conversionMap[sourceMesh.uniqueId];
                const replicatedSource = typeof replicatedSourceId === "number" ? storeMap[replicatedSourceId] : sourceMesh;
                const replicatedInstancedNode = replicatedSource.createInstance(instancedNode.name);
                onNewCreated(instancedNode, replicatedInstancedNode);
            }
            else {
                // Mesh or TransformNode
                let canInstance = true;
                if (node.getClassName() === "TransformNode" ||
                    node.getClassName() === "Node" ||
                    node.skeleton ||
                    !node.getTotalVertices ||
                    node.getTotalVertices() === 0) {
                    // Transform nodes, skinned meshes, and meshes with no vertices can never be instanced!
                    canInstance = false;
                }
                else if (localOptions.doNotInstantiate) {
                    if (typeof localOptions.doNotInstantiate === "function") {
                        canInstance = !localOptions.doNotInstantiate(node);
                    }
                    else {
                        canInstance = !localOptions.doNotInstantiate;
                    }
                }
                const replicatedNode = canInstance ? node.createInstance(`instance of ${node.name}`) : node.clone(`Clone of ${node.name}`, null, true);
                if (!replicatedNode) {
                    throw new Error(`Could not clone or instantiate node on Asset Container ${node.name}`);
                }
                onNewCreated(node, replicatedNode);
            }
        }
        for (const s of this.skeletons) {
            if (localOptions.predicate && !localOptions.predicate(s)) {
                continue;
            }
            const clone = s.clone(nameFunction ? nameFunction(s.name) : "Clone of " + s.name);
            for (const m of this.meshes) {
                if (m.skeleton === s && !m.isAnInstance) {
                    const copy = storeMap[conversionMap[m.uniqueId]];
                    if (!copy || copy.isAnInstance) {
                        continue;
                    }
                    copy.skeleton = clone;
                    if (alreadySwappedSkeletons.indexOf(clone) !== -1) {
                        continue;
                    }
                    alreadySwappedSkeletons.push(clone);
                    // Check if bones are mesh linked
                    for (const bone of clone.bones) {
                        if (bone._linkedTransformNode) {
                            bone._linkedTransformNode = storeMap[conversionMap[bone._linkedTransformNode.uniqueId]];
                        }
                    }
                }
            }
            result.skeletons.push(clone);
        }
        for (const o of this.animationGroups) {
            if (localOptions.predicate && !localOptions.predicate(o)) {
                continue;
            }
            const clone = o.clone(nameFunction ? nameFunction(o.name) : "Clone of " + o.name, (oldTarget) => {
                const newTarget = storeMap[conversionMap[oldTarget.uniqueId]];
                return newTarget || oldTarget;
            });
            result.animationGroups.push(clone);
        }
        return result;
    }
    /**
     * Adds all the assets from the container to the scene.
     */
    addAllToScene() {
        if (this._wasAddedToScene) {
            return;
        }
        if (!this._isValidHierarchy()) {
            Tools.Warn("SceneSerializer.addAllToScene: The Asset Container hierarchy is not valid.");
        }
        this._wasAddedToScene = true;
        this.addToScene(null);
        if (this.environmentTexture) {
            this.scene.environmentTexture = this.environmentTexture;
        }
        for (const component of this.scene._serializableComponents) {
            component.addFromContainer(this);
        }
        this.scene.getEngine().onContextRestoredObservable.remove(this._onContextRestoredObserver);
        this._onContextRestoredObserver = null;
    }
    /**
     * Adds assets from the container to the scene.
     * @param predicate defines a predicate used to select which entity will be added (can be null)
     */
    addToScene(predicate = null) {
        const addedNodes = [];
        for (const o of this.cameras) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.addCamera(o);
            addedNodes.push(o);
        }
        for (const o of this.lights) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.addLight(o);
            addedNodes.push(o);
        }
        for (const o of this.meshes) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.addMesh(o);
            addedNodes.push(o);
        }
        for (const o of this.skeletons) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.addSkeleton(o);
        }
        for (const o of this.animations) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.addAnimation(o);
        }
        for (const o of this.animationGroups) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.addAnimationGroup(o);
        }
        for (const o of this.multiMaterials) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.addMultiMaterial(o);
        }
        for (const o of this.materials) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.addMaterial(o);
        }
        for (const o of this.morphTargetManagers) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.addMorphTargetManager(o);
        }
        for (const o of this.geometries) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.addGeometry(o);
        }
        for (const o of this.transformNodes) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.addTransformNode(o);
            addedNodes.push(o);
        }
        for (const o of this.actionManagers) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.addActionManager(o);
        }
        for (const o of this.textures) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.addTexture(o);
        }
        for (const o of this.reflectionProbes) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.addReflectionProbe(o);
        }
        // No more nodes added to scene after this line, so it's safe to make a "snapshot" of nodes
        if (addedNodes.length) {
            // build the nodeSet only if needed
            const nodeSet = new Set(this.scene.meshes);
            // benchmark shows Set constructor and Set.add have similar performance,
            // but using Set.add here avoids another allocate in scene.getNodes().
            for (const light of this.scene.lights) {
                nodeSet.add(light);
            }
            for (const camera of this.scene.cameras) {
                nodeSet.add(camera);
            }
            for (const transformNode of this.scene.transformNodes) {
                nodeSet.add(transformNode);
            }
            for (const skeleton of this.skeletons) {
                for (const bone of skeleton.bones) {
                    nodeSet.add(bone);
                }
            }
            for (const addedNode of addedNodes) {
                // If node was added to the scene, but parent is not in the scene, break the relationship
                if (addedNode.parent && !nodeSet.has(addedNode.parent)) {
                    // Use setParent to keep transform if possible
                    if (addedNode.setParent) {
                        addedNode.setParent(null);
                    }
                    else {
                        addedNode.parent = null;
                    }
                }
            }
        }
    }
    /**
     * Removes all the assets in the container from the scene
     */
    removeAllFromScene() {
        if (!this._isValidHierarchy()) {
            Tools.Warn("SceneSerializer.removeAllFromScene: The Asset Container hierarchy is not valid.");
        }
        this._wasAddedToScene = false;
        this.removeFromScene(null);
        if (this.environmentTexture === this.scene.environmentTexture) {
            this.scene.environmentTexture = null;
        }
        for (const component of this.scene._serializableComponents) {
            component.removeFromContainer(this);
        }
    }
    /**
     * Removes assets in the container from the scene
     * @param predicate defines a predicate used to select which entity will be added (can be null)
     */
    removeFromScene(predicate = null) {
        for (const o of this.cameras) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.removeCamera(o);
        }
        for (const o of this.lights) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.removeLight(o);
        }
        for (const o of this.meshes) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.removeMesh(o, true);
        }
        for (const o of this.skeletons) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.removeSkeleton(o);
        }
        for (const o of this.animations) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.removeAnimation(o);
        }
        for (const o of this.animationGroups) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.removeAnimationGroup(o);
        }
        for (const o of this.multiMaterials) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.removeMultiMaterial(o);
        }
        for (const o of this.materials) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.removeMaterial(o);
        }
        for (const o of this.morphTargetManagers) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.removeMorphTargetManager(o);
        }
        for (const o of this.geometries) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.removeGeometry(o);
        }
        for (const o of this.transformNodes) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.removeTransformNode(o);
        }
        for (const o of this.actionManagers) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.removeActionManager(o);
        }
        for (const o of this.textures) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.removeTexture(o);
        }
        for (const o of this.reflectionProbes) {
            if (predicate && !predicate(o)) {
                continue;
            }
            this.scene.removeReflectionProbe(o);
        }
    }
    /**
     * Disposes all the assets in the container
     */
    dispose() {
        const cameras = this.cameras.slice(0);
        for (const camera of cameras) {
            camera.dispose();
        }
        this.cameras.length = 0;
        const lights = this.lights.slice(0);
        for (const light of lights) {
            light.dispose();
        }
        this.lights.length = 0;
        const meshes = this.meshes.slice(0);
        for (const mesh of meshes) {
            mesh.dispose();
        }
        this.meshes.length = 0;
        const skeletons = this.skeletons.slice(0);
        for (const skeleton of skeletons) {
            skeleton.dispose();
        }
        this.skeletons.length = 0;
        const animationGroups = this.animationGroups.slice(0);
        for (const animationGroup of animationGroups) {
            animationGroup.dispose();
        }
        this.animationGroups.length = 0;
        const multiMaterials = this.multiMaterials.slice(0);
        for (const multiMaterial of multiMaterials) {
            multiMaterial.dispose();
        }
        this.multiMaterials.length = 0;
        const materials = this.materials.slice(0);
        for (const material of materials) {
            material.dispose();
        }
        this.materials.length = 0;
        const geometries = this.geometries.slice(0);
        for (const geometry of geometries) {
            geometry.dispose();
        }
        this.geometries.length = 0;
        const transformNodes = this.transformNodes.slice(0);
        for (const transformNode of transformNodes) {
            transformNode.dispose();
        }
        this.transformNodes.length = 0;
        const actionManagers = this.actionManagers.slice(0);
        for (const actionManager of actionManagers) {
            actionManager.dispose();
        }
        this.actionManagers.length = 0;
        const textures = this.textures.slice(0);
        for (const texture of textures) {
            texture.dispose();
        }
        this.textures.length = 0;
        const reflectionProbes = this.reflectionProbes.slice(0);
        for (const reflectionProbe of reflectionProbes) {
            reflectionProbe.dispose();
        }
        this.reflectionProbes.length = 0;
        const morphTargetManagers = this.morphTargetManagers.slice(0);
        for (const morphTargetManager of morphTargetManagers) {
            morphTargetManager.dispose();
        }
        this.morphTargetManagers.length = 0;
        if (this.environmentTexture) {
            this.environmentTexture.dispose();
            this.environmentTexture = null;
        }
        for (const component of this.scene._serializableComponents) {
            component.removeFromContainer(this, true);
        }
        if (this._onContextRestoredObserver) {
            this.scene.getEngine().onContextRestoredObservable.remove(this._onContextRestoredObserver);
            this._onContextRestoredObserver = null;
        }
    }
    _moveAssets(sourceAssets, targetAssets, keepAssets) {
        if (!sourceAssets || !targetAssets) {
            return;
        }
        for (const asset of sourceAssets) {
            let move = true;
            if (keepAssets) {
                for (const keepAsset of keepAssets) {
                    if (asset === keepAsset) {
                        move = false;
                        break;
                    }
                }
            }
            if (move) {
                targetAssets.push(asset);
                asset._parentContainer = this;
            }
        }
    }
    /**
     * Removes all the assets contained in the scene and adds them to the container.
     * @param keepAssets Set of assets to keep in the scene. (default: empty)
     */
    moveAllFromScene(keepAssets) {
        this._wasAddedToScene = false;
        if (keepAssets === undefined) {
            keepAssets = new KeepAssets();
        }
        for (const key in this) {
            if (Object.prototype.hasOwnProperty.call(this, key)) {
                this[key] = this[key] || (key === "_environmentTexture" ? null : []);
                this._moveAssets(this.scene[key], this[key], keepAssets[key]);
            }
        }
        this.environmentTexture = this.scene.environmentTexture;
        this.removeAllFromScene();
    }
    /**
     * Adds all meshes in the asset container to a root mesh that can be used to position all the contained meshes. The root mesh is then added to the front of the meshes in the assetContainer.
     * @returns the root mesh
     */
    createRootMesh() {
        const rootMesh = new Mesh("assetContainerRootMesh", this.scene);
        for (const m of this.meshes) {
            if (!m.parent) {
                rootMesh.addChild(m);
            }
        }
        this.meshes.unshift(rootMesh);
        return rootMesh;
    }
    /**
     * Merge animations (direct and animation groups) from this asset container into a scene
     * @param scene is the instance of BABYLON.Scene to append to (default: last created scene)
     * @param animatables set of animatables to retarget to a node from the scene
     * @param targetConverter defines a function used to convert animation targets from the asset container to the scene (default: search node by name)
     * @returns an array of the new AnimationGroup added to the scene (empty array if none)
     */
    mergeAnimationsTo(scene = EngineStore.LastCreatedScene, animatables, targetConverter = null) {
        if (!scene) {
            Logger.Error("No scene available to merge animations to");
            return [];
        }
        const _targetConverter = targetConverter
            ? targetConverter
            : (target) => {
                let node = null;
                const targetProperty = target.animations.length ? target.animations[0].targetProperty : "";
                /*
            BabylonJS adds special naming to targets that are children of nodes.
            This name attempts to remove that special naming to get the parent nodes name in case the target
            can't be found in the node tree

            Ex: Torso_primitive0 likely points to a Mesh primitive. We take away primitive0 and are left with "Torso" which is the name
            of the primitive's parent.
        */
                const name = target.name.split(".").join("").split("_primitive")[0];
                switch (targetProperty) {
                    case "position":
                    case "rotationQuaternion":
                        node = scene.getTransformNodeByName(target.name) || scene.getTransformNodeByName(name);
                        break;
                    case "influence":
                        node = scene.getMorphTargetByName(target.name) || scene.getMorphTargetByName(name);
                        break;
                    default:
                        node = scene.getNodeByName(target.name) || scene.getNodeByName(name);
                }
                return node;
            };
        // Copy new node animations
        const nodesInAC = this.getNodes();
        for (const nodeInAC of nodesInAC) {
            const nodeInScene = _targetConverter(nodeInAC);
            if (nodeInScene !== null) {
                // Remove old animations with same target property as a new one
                for (const animationInAC of nodeInAC.animations) {
                    // Doing treatment on an array for safety measure
                    const animationsWithSameProperty = nodeInScene.animations.filter((animationInScene) => {
                        return animationInScene.targetProperty === animationInAC.targetProperty;
                    });
                    for (const animationWithSameProperty of animationsWithSameProperty) {
                        const index = nodeInScene.animations.indexOf(animationWithSameProperty, 0);
                        if (index > -1) {
                            nodeInScene.animations.splice(index, 1);
                        }
                    }
                }
                // Append new animations
                nodeInScene.animations = nodeInScene.animations.concat(nodeInAC.animations);
            }
        }
        const newAnimationGroups = [];
        // Copy new animation groups
        const animationGroups = this.animationGroups.slice();
        for (const animationGroupInAC of animationGroups) {
            // Clone the animation group and all its animatables
            newAnimationGroups.push(animationGroupInAC.clone(animationGroupInAC.name, _targetConverter));
            // Remove animatables related to the asset container
            for (const animatable of animationGroupInAC.animatables) {
                animatable.stop();
            }
        }
        // Retarget animatables
        for (const animatable of animatables) {
            const target = _targetConverter(animatable.target);
            if (target) {
                // Clone the animatable and retarget it
                scene.beginAnimation(target, animatable.fromFrame, animatable.toFrame, animatable.loopAnimation, animatable.speedRatio, animatable.onAnimationEnd ? animatable.onAnimationEnd : undefined, undefined, true, undefined, animatable.onAnimationLoop ? animatable.onAnimationLoop : undefined);
                // Stop animation for the target in the asset container
                scene.stopAnimation(animatable.target);
            }
        }
        return newAnimationGroups;
    }
    /**
     * @since 6.15.0
     * This method checks for any node that has no parent
     * and is not in the rootNodes array, and adds the node
     * there, if so.
     */
    populateRootNodes() {
        this.rootNodes.length = 0;
        for (const m of this.meshes) {
            if (!m.parent && this.rootNodes.indexOf(m) === -1) {
                this.rootNodes.push(m);
            }
        }
        for (const t of this.transformNodes) {
            if (!t.parent && this.rootNodes.indexOf(t) === -1) {
                this.rootNodes.push(t);
            }
        }
        for (const l of this.lights) {
            if (!l.parent && this.rootNodes.indexOf(l) === -1) {
                this.rootNodes.push(l);
            }
        }
        for (const c of this.cameras) {
            if (!c.parent && this.rootNodes.indexOf(c) === -1) {
                this.rootNodes.push(c);
            }
        }
    }
    /**
     * @since 6.26.0
     * Given a root asset, this method will traverse its hierarchy and add it, its children and any materials/skeletons to the container.
     * @param root root node
     */
    addAllAssetsToContainer(root) {
        if (!root) {
            return;
        }
        const nodesToVisit = [];
        const visitedNodes = new Set();
        nodesToVisit.push(root);
        while (nodesToVisit.length > 0) {
            const nodeToVisit = nodesToVisit.pop();
            if (nodeToVisit instanceof Mesh) {
                if (nodeToVisit.geometry && this.geometries.indexOf(nodeToVisit.geometry) === -1) {
                    this.geometries.push(nodeToVisit.geometry);
                }
                this.meshes.push(nodeToVisit);
            }
            else if (nodeToVisit instanceof InstancedMesh) {
                this.meshes.push(nodeToVisit);
            }
            else if (nodeToVisit instanceof TransformNode) {
                this.transformNodes.push(nodeToVisit);
            }
            else if (nodeToVisit instanceof Light) {
                this.lights.push(nodeToVisit);
            }
            else if (nodeToVisit instanceof Camera) {
                this.cameras.push(nodeToVisit);
            }
            if (nodeToVisit instanceof AbstractMesh) {
                if (nodeToVisit.material && this.materials.indexOf(nodeToVisit.material) === -1) {
                    this.materials.push(nodeToVisit.material);
                    for (const texture of nodeToVisit.material.getActiveTextures()) {
                        if (this.textures.indexOf(texture) === -1) {
                            this.textures.push(texture);
                        }
                    }
                }
                if (nodeToVisit.skeleton && this.skeletons.indexOf(nodeToVisit.skeleton) === -1) {
                    this.skeletons.push(nodeToVisit.skeleton);
                }
                if (nodeToVisit.morphTargetManager && this.morphTargetManagers.indexOf(nodeToVisit.morphTargetManager) === -1) {
                    this.morphTargetManagers.push(nodeToVisit.morphTargetManager);
                }
            }
            for (const child of nodeToVisit.getChildren()) {
                if (!visitedNodes.has(child)) {
                    nodesToVisit.push(child);
                }
            }
            visitedNodes.add(nodeToVisit);
        }
        this.populateRootNodes();
    }
    /**
     * Get from a list of objects by tags
     * @param list the list of objects to use
     * @param tagsQuery the query to use
     * @param filter a predicate to filter for tags
     * @returns
     */
    _getByTags(list, tagsQuery, filter) {
        if (tagsQuery === undefined) {
            // returns the complete list (could be done with Tags.MatchesQuery but no need to have a for-loop here)
            return list;
        }
        const listByTags = [];
        for (const i in list) {
            const item = list[i];
            if (Tags && Tags.MatchesQuery(item, tagsQuery) && (!filter || filter(item))) {
                listByTags.push(item);
            }
        }
        return listByTags;
    }
    /**
     * Get a list of meshes by tags
     * @param tagsQuery defines the tags query to use
     * @param filter defines a predicate used to filter results
     * @returns an array of Mesh
     */
    getMeshesByTags(tagsQuery, filter) {
        return this._getByTags(this.meshes, tagsQuery, filter);
    }
    /**
     * Get a list of cameras by tags
     * @param tagsQuery defines the tags query to use
     * @param filter defines a predicate used to filter results
     * @returns an array of Camera
     */
    getCamerasByTags(tagsQuery, filter) {
        return this._getByTags(this.cameras, tagsQuery, filter);
    }
    /**
     * Get a list of lights by tags
     * @param tagsQuery defines the tags query to use
     * @param filter defines a predicate used to filter results
     * @returns an array of Light
     */
    getLightsByTags(tagsQuery, filter) {
        return this._getByTags(this.lights, tagsQuery, filter);
    }
    /**
     * Get a list of materials by tags
     * @param tagsQuery defines the tags query to use
     * @param filter defines a predicate used to filter results
     * @returns an array of Material
     */
    getMaterialsByTags(tagsQuery, filter) {
        return this._getByTags(this.materials, tagsQuery, filter).concat(this._getByTags(this.multiMaterials, tagsQuery, filter));
    }
    /**
     * Get a list of transform nodes by tags
     * @param tagsQuery defines the tags query to use
     * @param filter defines a predicate used to filter results
     * @returns an array of TransformNode
     */
    getTransformNodesByTags(tagsQuery, filter) {
        return this._getByTags(this.transformNodes, tagsQuery, filter);
    }
}

export { AssetContainer as A };
//# sourceMappingURL=assetContainer-7GPvH2S_.js.map
