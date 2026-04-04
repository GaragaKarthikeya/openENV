const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./flowGraphPlayAnimationBlock-BkKzjOFT.js","./declarationMapper-eK_q7dJF.js","./index-CxR0Yf_7.js","./index-BvBk1Iap.js","./index-CrbFh6qt.css","./animationGroup-CGqF01WM.js","./bone-C6xIrJX4.js","./objectModelMapping-DnUo8NMz.js","./flowGraphStopAnimationBlock-Cf1LB0Bk.js","./flowGraphPauseAnimationBlock-B9i08t1Q.js","./flowGraphInterpolationBlock-a-o18QBY.js","./flowGraphSceneReadyEventBlock-vtf4yMOr.js","./flowGraphSceneTickEventBlock-DRNAn94I.js","./flowGraphSendCustomEventBlock-6rVFn3mr.js","./flowGraphReceiveCustomEventBlock-CC4v1Q-T.js","./flowGraphMeshPickEventBlock-BaFqgJHG.js","./flowGraphMathBlocks-CBHd7jx3.js","./flowGraphBinaryOperationBlock-DJwjQkLd.js","./flowGraphCachedOperationBlock-HVWAtkJN.js","./flowGraphUnaryOperationBlock-BPV858Ax.js","./flowGraphVectorMathBlocks-BMWlT9Au.js","./flowGraphMatrixMathBlocks-CVA57ClY.js","./flowGraphBranchBlock-DiLZOxxs.js","./flowGraphSetDelayBlock-BH-Ttv2A.js","./flowGraphCancelDelayBlock-DdnCJIHp.js","./flowGraphCounterBlock-CHfEWEKo.js","./flowGraphDebounceBlock-B7BN833I.js","./flowGraphThrottleBlock-xEOLL07m.js","./flowGraphDoNBlock-BsIGVnlM.js","./flowGraphFlipFlopBlock-CvA24zJ2.js","./flowGraphForLoopBlock-objpPanS.js","./flowGraphMultiGateBlock-DiPiWOHH.js","./flowGraphSequenceBlock-B0Rrk3RL.js","./flowGraphSwitchBlock-twUIuJlv.js","./flowGraphWaitAllBlock-m7YQzoZP.js","./flowGraphWhileLoopBlock-DzQfdud0.js","./flowGraphConsoleLogBlock-C6T6-DAQ.js","./flowGraphConditionalDataBlock-C-UNv5MV.js","./flowGraphConstantBlock-Cd6_VUbG.js","./flowGraphTransformCoordinatesSystemBlock-Ci41pAjS.js","./flowGraphGetAssetBlock-eRHQvRSt.js","./flowGraphGetPropertyBlock-B2jLqZtA.js","./flowGraphSetPropertyBlock-C412YpUC.js","./flowGraphGetVariableBlock-D91Cgc9v.js","./flowGraphSetVariableBlock-Bp8960gb.js","./flowGraphJsonPointerParserBlock-BkiHU-aD.js","./flowGraphMathCombineExtractBlocks-qEm2nMP2.js","./flowGraphTypeToTypeBlocks-BCIt2CE0.js","./flowGraphEasingBlock-DPpBHrfT.js","./flowGraphBezierCurveEasingBlock-Q5r3yjWs.js","./flowGraphPointerOverEventBlock-DMmP4NMG.js","./flowGraphPointerOutEventBlock-BTSYlAKI.js","./flowGraphContextBlock-D4tc8TQP.js","./flowGraphArrayIndexBlock-NJcLU5dO.js","./flowGraphCodeExecutionBlock-D2nzjSIh.js","./flowGraphIndexOfBlock-EHh5o3oq.js","./flowGraphFunctionReferenceBlock-ByGpsFPE.js","./flowGraphDataSwitchBlock-DSObniwd.js","./flowGraphGLTFDataProvider-BF36Cs2j.js"])))=>i.map(i=>d[i]);
import { af as __vitePreload } from './index-BvBk1Iap.js';
import { M as Matrix, ah as Vector2, V as Vector3, ai as Vector4, ao as Quaternion, C as Color3, b9 as Color4, L as Logger, h as RandomGUID, O as Observable, _ as __decorate, s as serialize, R as RegisterClass, aQ as PointerEventTypes, aG as unregisterGLTFExtension, aH as registerGLTFExtension } from './index-CxR0Yf_7.js';
import { n as FlowGraphMatrix2D, p as FlowGraphMatrix3D, F as FlowGraphInteger, g as getRichTypeByFlowGraphType, r as getMappingForDeclaration, s as getMappingForFullOperationName } from './declarationMapper-eK_q7dJF.js';
import { G as GetPathToObjectConverter, A as AddObjectAccessorToKey } from './objectModelMapping-DnUo8NMz.js';

function IsMeshClassName(className) {
    return (className === "Mesh" ||
        className === "AbstractMesh" ||
        className === "GroundMesh" ||
        className === "InstanceMesh" ||
        className === "LinesMesh" ||
        className === "GoldbergMesh" ||
        className === "GreasedLineMesh" ||
        className === "TrailMesh");
}
function IsVectorClassName(className) {
    return (className === "Vector2" /* FlowGraphTypes.Vector2 */ ||
        className === "Vector3" /* FlowGraphTypes.Vector3 */ ||
        className === "Vector4" /* FlowGraphTypes.Vector4 */ ||
        className === "Quaternion" /* FlowGraphTypes.Quaternion */ ||
        className === "Color3" /* FlowGraphTypes.Color3 */ ||
        className === "Color4" /* FlowGraphTypes.Color4 */);
}
function IsMatrixClassName(className) {
    return className === "Matrix" /* FlowGraphTypes.Matrix */ || className === "Matrix2D" /* FlowGraphTypes.Matrix2D */ || className === "Matrix3D" /* FlowGraphTypes.Matrix3D */;
}
function IsAnimationGroupClassName(className) {
    return className === "AnimationGroup";
}
function ParseVector(className, value, flipHandedness = false) {
    if (className === "Vector2" /* FlowGraphTypes.Vector2 */) {
        return Vector2.FromArray(value);
    }
    else if (className === "Vector3" /* FlowGraphTypes.Vector3 */) {
        if (flipHandedness) {
            value[2] *= -1;
        }
        return Vector3.FromArray(value);
    }
    else if (className === "Vector4" /* FlowGraphTypes.Vector4 */) {
        return Vector4.FromArray(value);
    }
    else if (className === "Quaternion" /* FlowGraphTypes.Quaternion */) {
        if (flipHandedness) {
            value[2] *= -1;
            value[3] *= -1;
        }
        return Quaternion.FromArray(value);
    }
    else if (className === "Color3" /* FlowGraphTypes.Color3 */) {
        return new Color3(value[0], value[1], value[2]);
    }
    else if (className === "Color4" /* FlowGraphTypes.Color4 */) {
        return new Color4(value[0], value[1], value[2], value[3]);
    }
    else {
        throw new Error(`Unknown vector class name ${className}`);
    }
}
/**
 * The default function that serializes values in a context object to a serialization object
 * @param key the key where the value should be stored in the serialization object
 * @param value the value to store
 * @param serializationObject the object where the value will be stored
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
function defaultValueSerializationFunction(key, value, serializationObject) {
    const className = value?.getClassName?.() ?? "";
    if (IsVectorClassName(className) || IsMatrixClassName(className)) {
        serializationObject[key] = {
            value: value.asArray(),
            className,
        };
    }
    else if (className === "FlowGraphInteger" /* FlowGraphTypes.Integer */) {
        serializationObject[key] = {
            value: value.value,
            className,
        };
    }
    else {
        if (className && (value.id || value.name)) {
            serializationObject[key] = {
                id: value.id,
                name: value.name,
                className,
            };
        }
        else {
            // only if it is not an object
            if (typeof value !== "object") {
                serializationObject[key] = value;
            }
            else {
                throw new Error(`Could not serialize value ${value}`);
            }
        }
    }
}
/**
 * The default function that parses values stored in a serialization object
 * @param key the key to the value that will be parsed
 * @param serializationObject the object that will be parsed
 * @param assetsContainer the assets container that will be used to find the objects
 * @param scene
 * @returns
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
function defaultValueParseFunction(key, serializationObject, assetsContainer, scene) {
    const intermediateValue = serializationObject[key];
    let finalValue;
    const className = intermediateValue?.type ?? intermediateValue?.className;
    if (IsMeshClassName(className)) {
        let nodes = scene.meshes.filter((m) => (intermediateValue.id ? m.id === intermediateValue.id : m.name === intermediateValue.name));
        if (nodes.length === 0) {
            nodes = scene.transformNodes.filter((m) => (intermediateValue.id ? m.id === intermediateValue.id : m.name === intermediateValue.name));
        }
        finalValue = intermediateValue.uniqueId ? nodes.find((m) => m.uniqueId === intermediateValue.uniqueId) : nodes[0];
    }
    else if (IsVectorClassName(className)) {
        finalValue = ParseVector(className, intermediateValue.value);
    }
    else if (IsAnimationGroupClassName(className)) {
        // do not use the scene.getAnimationGroupByName because it is possible that two AGs will have the same name
        const ags = scene.animationGroups.filter((ag) => ag.name === intermediateValue.name);
        // uniqueId changes on each load. this is used for the glTF loader, that uses serialization after the scene was loaded.
        finalValue = ags.length === 1 ? ags[0] : ags.find((ag) => ag.uniqueId === intermediateValue.uniqueId);
    }
    else if (className === "Matrix" /* FlowGraphTypes.Matrix */) {
        finalValue = Matrix.FromArray(intermediateValue.value);
    }
    else if (className === "Matrix2D" /* FlowGraphTypes.Matrix2D */) {
        finalValue = new FlowGraphMatrix2D(intermediateValue.value);
    }
    else if (className === "Matrix3D" /* FlowGraphTypes.Matrix3D */) {
        finalValue = new FlowGraphMatrix3D(intermediateValue.value);
    }
    else if (className === "FlowGraphInteger" /* FlowGraphTypes.Integer */) {
        finalValue = FlowGraphInteger.FromValue(intermediateValue.value);
    }
    else if (className === "number" /* FlowGraphTypes.Number */ || className === "string" /* FlowGraphTypes.String */ || className === "boolean" /* FlowGraphTypes.Boolean */) {
        finalValue = intermediateValue.value[0];
    }
    else if (intermediateValue && intermediateValue.value !== undefined) {
        finalValue = intermediateValue.value;
    }
    else {
        if (Array.isArray(intermediateValue)) {
            // configuration data of an event
            finalValue = intermediateValue.reduce((acc, val) => {
                if (!val.eventData) {
                    return acc;
                }
                acc[val.id] = {
                    type: getRichTypeByFlowGraphType(val.type),
                };
                if (typeof val.value !== "undefined") {
                    acc[val.id].value = defaultValueParseFunction("value", val, assetsContainer, scene);
                }
                return acc;
            }, {});
        }
        else {
            finalValue = intermediateValue;
        }
    }
    return finalValue;
}
/**
 * Given a name of a flow graph block class, return if this
 * class needs to be created with a path converter. Used in
 * parsing.
 * @param className the name of the flow graph block class
 * @returns a boolean indicating if the class needs a path converter
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
function needsPathConverter(className) {
    // I am not using the ClassName property here because it was causing a circular dependency
    // that jest didn't like!
    return className === "FlowGraphJsonPointerParserBlock" /* FlowGraphBlockNames.JsonPointerParser */;
}

/**
 * The type of the assets that flow graph supports
 */
var FlowGraphAssetType;
(function (FlowGraphAssetType) {
    FlowGraphAssetType["Animation"] = "Animation";
    FlowGraphAssetType["AnimationGroup"] = "AnimationGroup";
    FlowGraphAssetType["Mesh"] = "Mesh";
    FlowGraphAssetType["Material"] = "Material";
    FlowGraphAssetType["Camera"] = "Camera";
    FlowGraphAssetType["Light"] = "Light";
    // Further asset types will be added here when needed.
})(FlowGraphAssetType || (FlowGraphAssetType = {}));
/**
 * Returns the asset with the given index and type from the assets context.
 * @param assetsContext The assets context to get the asset from
 * @param type The type of the asset
 * @param index The index of the asset
 * @param useIndexAsUniqueId If set to true, instead of the index in the array it will search for the unique id of the asset.
 * @returns The asset or null if not found
 */
function GetFlowGraphAssetWithType(assetsContext, type, index, useIndexAsUniqueId) {
    switch (type) {
        case "Animation" /* FlowGraphAssetType.Animation */:
            return useIndexAsUniqueId
                ? (assetsContext.animations.find((a) => a.uniqueId === index) ?? null)
                : (assetsContext.animations[index] ?? null);
        case "AnimationGroup" /* FlowGraphAssetType.AnimationGroup */:
            return useIndexAsUniqueId
                ? (assetsContext.animationGroups.find((a) => a.uniqueId === index) ?? null)
                : (assetsContext.animationGroups[index] ?? null);
        case "Mesh" /* FlowGraphAssetType.Mesh */:
            return useIndexAsUniqueId
                ? (assetsContext.meshes.find((a) => a.uniqueId === index) ?? null)
                : (assetsContext.meshes[index] ?? null);
        case "Material" /* FlowGraphAssetType.Material */:
            return useIndexAsUniqueId
                ? (assetsContext.materials.find((a) => a.uniqueId === index) ?? null)
                : (assetsContext.materials[index] ?? null);
        case "Camera" /* FlowGraphAssetType.Camera */:
            return useIndexAsUniqueId
                ? (assetsContext.cameras.find((a) => a.uniqueId === index) ?? null)
                : (assetsContext.cameras[index] ?? null);
        case "Light" /* FlowGraphAssetType.Light */:
            return useIndexAsUniqueId
                ? (assetsContext.lights.find((a) => a.uniqueId === index) ?? null)
                : (assetsContext.lights[index] ?? null);
        default:
            return null;
    }
}

var FlowGraphAction;
(function (FlowGraphAction) {
    FlowGraphAction["ExecuteBlock"] = "ExecuteBlock";
    FlowGraphAction["ExecuteEvent"] = "ExecuteEvent";
    FlowGraphAction["TriggerConnection"] = "TriggerConnection";
    FlowGraphAction["ContextVariableSet"] = "ContextVariableSet";
    FlowGraphAction["GlobalVariableSet"] = "GlobalVariableSet";
    FlowGraphAction["GlobalVariableDelete"] = "GlobalVariableDelete";
    FlowGraphAction["GlobalVariableGet"] = "GlobalVariableGet";
    FlowGraphAction["AddConnection"] = "AddConnection";
    FlowGraphAction["GetConnectionValue"] = "GetConnectionValue";
    FlowGraphAction["SetConnectionValue"] = "SetConnectionValue";
    FlowGraphAction["ActivateSignal"] = "ActivateSignal";
    FlowGraphAction["ContextVariableGet"] = "ContextVariableGet";
})(FlowGraphAction || (FlowGraphAction = {}));
/**
 * This class will be responsible of logging the flow graph activity.
 * Note that using this class might reduce performance, as it will log every action, according to the configuration.
 * It attaches to a flow graph and uses meta-programming to replace the methods of the flow graph to add logging abilities.
 */
class FlowGraphLogger {
    constructor() {
        /**
         * Whether to log to the console.
         */
        this.logToConsole = false;
        /**
         * The log cache of the flow graph.
         * Each item is a logged item, in order of execution.
         */
        this.log = [];
    }
    addLogItem(item) {
        if (!item.time) {
            item.time = Date.now();
        }
        this.log.push(item);
        if (this.logToConsole) {
            const value = item.payload?.value;
            if (typeof value === "object" && value.getClassName) {
                Logger.Log(`[FGLog] ${item.className}:${item.uniqueId.split("-")[0]} ${item.action} - ${JSON.stringify(value.getClassName())}: ${value.toString()}`);
            }
            else {
                Logger.Log(`[FGLog] ${item.className}:${item.uniqueId.split("-")[0]} ${item.action} - ${JSON.stringify(item.payload)}`);
            }
        }
    }
    getItemsOfType(action) {
        return this.log.filter((i) => i.action === action);
    }
}

/**
 * The context represents the current state and execution of the flow graph.
 * It contains both user-defined variables, which are derived from
 * a more general variable definition, and execution variables that
 * are set by the blocks.
 */
class FlowGraphContext {
    /**
     * Enable logging on this context
     */
    get enableLogging() {
        return this._enableLogging;
    }
    set enableLogging(value) {
        if (this._enableLogging === value) {
            return;
        }
        this._enableLogging = value;
        if (this._enableLogging) {
            this.logger = new FlowGraphLogger();
            this.logger.logToConsole = true;
        }
        else {
            this.logger = null;
        }
    }
    constructor(params) {
        /**
         * A randomly generated GUID for each context.
         */
        this.uniqueId = RandomGUID();
        /**
         * These are the variables defined by a user.
         */
        this._userVariables = {};
        /**
         * These are the variables set by the blocks.
         */
        this._executionVariables = {};
        /**
         * A context-specific global variables, available to all blocks in the context.
         */
        this._globalContextVariables = {};
        /**
         * These are the values for the data connection points
         */
        this._connectionValues = {};
        /**
         * These are blocks that have currently pending tasks/listeners that need to be cleaned up.
         */
        this._pendingBlocks = [];
        /**
         * A monotonically increasing ID for each execution.
         * Incremented for every block executed.
         */
        this._executionId = 0;
        /**
         * Observable that is triggered when a node is executed.
         */
        this.onNodeExecutedObservable = new Observable();
        /**
         * Whether to treat data as right-handed.
         * This is used when serializing data from a right-handed system, while running the context in a left-handed system, for example in glTF parsing.
         * Default is false.
         */
        this.treatDataAsRightHanded = false;
        this._enableLogging = false;
        this._configuration = params;
        this.assetsContext = params.assetsContext ?? params.scene;
    }
    /**
     * Check if a user-defined variable is defined.
     * @param name the name of the variable
     * @returns true if the variable is defined
     */
    hasVariable(name) {
        return name in this._userVariables;
    }
    /**
     * Set a user-defined variable.
     * @param name the name of the variable
     * @param value the value of the variable
     */
    setVariable(name, value) {
        this._userVariables[name] = value;
        this.logger?.addLogItem({
            time: Date.now(),
            className: this.getClassName(),
            uniqueId: this.uniqueId,
            action: "ContextVariableSet" /* FlowGraphAction.ContextVariableSet */,
            payload: {
                name,
                value,
            },
        });
    }
    /**
     * Get an assets from the assets context based on its type and index in the array
     * @param type The type of the asset
     * @param index The index of the asset
     * @returns The asset or null if not found
     */
    getAsset(type, index) {
        return GetFlowGraphAssetWithType(this.assetsContext, type, index);
    }
    /**
     * Get a user-defined variable.
     * @param name the name of the variable
     * @returns the value of the variable
     */
    getVariable(name) {
        this.logger?.addLogItem({
            time: Date.now(),
            className: this.getClassName(),
            uniqueId: this.uniqueId,
            action: "ContextVariableGet" /* FlowGraphAction.ContextVariableGet */,
            payload: {
                name,
                value: this._userVariables[name],
            },
        });
        return this._userVariables[name];
    }
    /**
     * Gets all user variables map
     */
    get userVariables() {
        return this._userVariables;
    }
    /**
     * Get the scene that the context belongs to.
     * @returns the scene
     */
    getScene() {
        return this._configuration.scene;
    }
    _getUniqueIdPrefixedName(obj, name) {
        return `${obj.uniqueId}_${name}`;
    }
    /**
     * @internal
     * @param name name of the variable
     * @param defaultValue default value to return if the variable is not defined
     * @returns the variable value or the default value if the variable is not defined
     */
    _getGlobalContextVariable(name, defaultValue) {
        this.logger?.addLogItem({
            time: Date.now(),
            className: this.getClassName(),
            uniqueId: this.uniqueId,
            action: "GlobalVariableGet" /* FlowGraphAction.GlobalVariableGet */,
            payload: {
                name,
                defaultValue,
                possibleValue: this._globalContextVariables[name],
            },
        });
        if (this._hasGlobalContextVariable(name)) {
            return this._globalContextVariables[name];
        }
        else {
            return defaultValue;
        }
    }
    /**
     * Set a global context variable
     * @internal
     * @param name the name of the variable
     * @param value the value of the variable
     */
    _setGlobalContextVariable(name, value) {
        this.logger?.addLogItem({
            time: Date.now(),
            className: this.getClassName(),
            uniqueId: this.uniqueId,
            action: "GlobalVariableSet" /* FlowGraphAction.GlobalVariableSet */,
            payload: { name, value },
        });
        this._globalContextVariables[name] = value;
    }
    /**
     * Delete a global context variable
     * @internal
     * @param name the name of the variable
     */
    _deleteGlobalContextVariable(name) {
        this.logger?.addLogItem({
            time: Date.now(),
            className: this.getClassName(),
            uniqueId: this.uniqueId,
            action: "GlobalVariableDelete" /* FlowGraphAction.GlobalVariableDelete */,
            payload: { name },
        });
        delete this._globalContextVariables[name];
    }
    /**
     * Check if a global context variable is defined
     * @internal
     * @param name the name of the variable
     * @returns true if the variable is defined
     */
    _hasGlobalContextVariable(name) {
        return name in this._globalContextVariables;
    }
    /**
     * Set an internal execution variable
     * @internal
     * @param name
     * @param value
     */
    _setExecutionVariable(block, name, value) {
        this._executionVariables[this._getUniqueIdPrefixedName(block, name)] = value;
    }
    /**
     * Get an internal execution variable
     * @internal
     * @param name
     * @returns
     */
    _getExecutionVariable(block, name, defaultValue) {
        if (this._hasExecutionVariable(block, name)) {
            return this._executionVariables[this._getUniqueIdPrefixedName(block, name)];
        }
        else {
            return defaultValue;
        }
    }
    /**
     * Delete an internal execution variable
     * @internal
     * @param block
     * @param name
     */
    _deleteExecutionVariable(block, name) {
        delete this._executionVariables[this._getUniqueIdPrefixedName(block, name)];
    }
    /**
     * Check if an internal execution variable is defined
     * @internal
     * @param block
     * @param name
     * @returns
     */
    _hasExecutionVariable(block, name) {
        return this._getUniqueIdPrefixedName(block, name) in this._executionVariables;
    }
    /**
     * Check if a connection value is defined
     * @internal
     * @param connectionPoint
     * @returns
     */
    _hasConnectionValue(connectionPoint) {
        return connectionPoint.uniqueId in this._connectionValues;
    }
    /**
     * Set a connection value
     * @internal
     * @param connectionPoint
     * @param value
     */
    _setConnectionValue(connectionPoint, value) {
        this._connectionValues[connectionPoint.uniqueId] = value;
        this.logger?.addLogItem({
            time: Date.now(),
            className: this.getClassName(),
            uniqueId: this.uniqueId,
            action: "SetConnectionValue" /* FlowGraphAction.SetConnectionValue */,
            payload: {
                connectionPointId: connectionPoint.uniqueId,
                value,
            },
        });
    }
    /**
     * Set a connection value by key
     * @internal
     * @param key the key of the connection value
     * @param value the value of the connection
     */
    _setConnectionValueByKey(key, value) {
        this._connectionValues[key] = value;
    }
    /**
     * Get a connection value
     * @internal
     * @param connectionPoint
     * @returns
     */
    _getConnectionValue(connectionPoint) {
        this.logger?.addLogItem({
            time: Date.now(),
            className: this.getClassName(),
            uniqueId: this.uniqueId,
            action: "GetConnectionValue" /* FlowGraphAction.GetConnectionValue */,
            payload: {
                connectionPointId: connectionPoint.uniqueId,
                value: this._connectionValues[connectionPoint.uniqueId],
            },
        });
        return this._connectionValues[connectionPoint.uniqueId];
    }
    /**
     * Get the configuration
     * @internal
     * @param name
     * @param value
     */
    get configuration() {
        return this._configuration;
    }
    /**
     * Check if there are any pending blocks in this context
     * @returns true if there are pending blocks
     */
    get hasPendingBlocks() {
        return this._pendingBlocks.length > 0;
    }
    /**
     * Add a block to the list of blocks that have pending tasks.
     * @internal
     * @param block
     */
    _addPendingBlock(block) {
        // check if block is already in the array
        if (this._pendingBlocks.includes(block)) {
            return;
        }
        this._pendingBlocks.push(block);
        // sort pending blocks by priority
        this._pendingBlocks.sort((a, b) => a.priority - b.priority);
    }
    /**
     * Remove a block from the list of blocks that have pending tasks.
     * @internal
     * @param block
     */
    _removePendingBlock(block) {
        const index = this._pendingBlocks.indexOf(block);
        if (index !== -1) {
            this._pendingBlocks.splice(index, 1);
        }
    }
    /**
     * Clear all pending blocks.
     * @internal
     */
    _clearPendingBlocks() {
        for (const block of this._pendingBlocks) {
            block._cancelPendingTasks(this);
        }
        this._pendingBlocks.length = 0;
    }
    /**
     * @internal
     * Function that notifies the node executed observable
     * @param node
     */
    _notifyExecuteNode(node) {
        this.onNodeExecutedObservable.notifyObservers(node);
        this.logger?.addLogItem({
            time: Date.now(),
            className: node.getClassName(),
            uniqueId: node.uniqueId,
            action: "ExecuteBlock" /* FlowGraphAction.ExecuteBlock */,
        });
    }
    _notifyOnTick(framePayload) {
        // set the values as global variables
        this._setGlobalContextVariable("timeSinceStart", framePayload.timeSinceStart);
        this._setGlobalContextVariable("deltaTime", framePayload.deltaTime);
        // iterate the pending blocks and run each one's onFrame function
        for (const block of this._pendingBlocks) {
            block._executeOnTick?.(this);
        }
    }
    /**
     * @internal
     */
    _increaseExecutionId() {
        this._executionId++;
    }
    /**
     * A monotonically increasing ID for each execution.
     * Incremented for every block executed.
     */
    get executionId() {
        return this._executionId;
    }
    /**
     * Serializes a context
     * @param serializationObject the object to write the values in
     * @param valueSerializationFunction a function to serialize complex values
     */
    serialize(serializationObject = {}, valueSerializationFunction = defaultValueSerializationFunction) {
        serializationObject.uniqueId = this.uniqueId;
        serializationObject._userVariables = {};
        for (const key in this._userVariables) {
            valueSerializationFunction(key, this._userVariables[key], serializationObject._userVariables);
        }
        serializationObject._connectionValues = {};
        for (const key in this._connectionValues) {
            valueSerializationFunction(key, this._connectionValues[key], serializationObject._connectionValues);
        }
        // serialize assets context, if not scene
        if (this.assetsContext !== this.getScene()) {
            serializationObject._assetsContext = {
                meshes: this.assetsContext.meshes.map((m) => m.id),
                materials: this.assetsContext.materials.map((m) => m.id),
                textures: this.assetsContext.textures.map((m) => m.name),
                animations: this.assetsContext.animations.map((m) => m.name),
                lights: this.assetsContext.lights.map((m) => m.id),
                cameras: this.assetsContext.cameras.map((m) => m.id),
                sounds: this.assetsContext.sounds?.map((m) => m.name),
                skeletons: this.assetsContext.skeletons.map((m) => m.id),
                particleSystems: this.assetsContext.particleSystems.map((m) => m.name),
                geometries: this.assetsContext.geometries.map((m) => m.id),
                multiMaterials: this.assetsContext.multiMaterials.map((m) => m.id),
                transformNodes: this.assetsContext.transformNodes.map((m) => m.id),
            };
        }
    }
    /**
     * @returns the class name of the object.
     */
    getClassName() {
        return "FlowGraphContext";
    }
}
__decorate([
    serialize()
], FlowGraphContext.prototype, "uniqueId", void 0);

/**
 * The type of a connection point - input or output.
 */
var FlowGraphConnectionType;
(function (FlowGraphConnectionType) {
    FlowGraphConnectionType[FlowGraphConnectionType["Input"] = 0] = "Input";
    FlowGraphConnectionType[FlowGraphConnectionType["Output"] = 1] = "Output";
})(FlowGraphConnectionType || (FlowGraphConnectionType = {}));
/**
 * The base connection class.
 */
class FlowGraphConnection {
    constructor(name, _connectionType, 
    /* @internal */ _ownerBlock) {
        this._ownerBlock = _ownerBlock;
        /** @internal */
        this._connectedPoint = [];
        /**
         * A uniquely identifying string for the connection.
         */
        this.uniqueId = RandomGUID();
        /**
         * Used for parsing connections.
         * @internal
         */
        // disable warning as this is used for parsing
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        this.connectedPointIds = [];
        this.name = name;
        this._connectionType = _connectionType;
    }
    /**
     * The type of the connection
     */
    get connectionType() {
        return this._connectionType;
    }
    /**
     * @internal
     * Override this to indicate if a point can connect to more than one point.
     */
    _isSingularConnection() {
        return true;
    }
    /**
     * Returns if a point is connected to any other point.
     * @returns boolean indicating if the point is connected.
     */
    isConnected() {
        return this._connectedPoint.length > 0;
    }
    /**
     * Connects two connections together.
     * @param point the connection to connect to.
     */
    connectTo(point) {
        if (this._connectionType === point._connectionType) {
            throw new Error(`Cannot connect two points of type ${this.connectionType}`);
        }
        if ((this._isSingularConnection() && this._connectedPoint.length > 0) || (point._isSingularConnection() && point._connectedPoint.length > 0)) {
            throw new Error("Max number of connections for point reached");
        }
        this._connectedPoint.push(point);
        point._connectedPoint.push(this);
    }
    /**
     * Disconnects two connections.
     * @param point the connection to disconnect from.
     * @param removeFromLocal if true, the connection will be removed from the local connection list.
     */
    disconnectFrom(point, removeFromLocal = true) {
        const indexLocal = this._connectedPoint.indexOf(point);
        const indexConnected = point._connectedPoint.indexOf(this);
        if (indexLocal === -1 || indexConnected === -1) {
            return;
        }
        if (removeFromLocal) {
            this._connectedPoint.splice(indexLocal, 1);
        }
        point._connectedPoint.splice(indexConnected, 1);
    }
    /**
     * Disconnects all connected points.
     */
    disconnectFromAll() {
        for (const point of this._connectedPoint) {
            this.disconnectFrom(point, false);
        }
        this._connectedPoint.length = 0;
    }
    dispose() {
        for (const point of this._connectedPoint) {
            this.disconnectFrom(point);
        }
    }
    /**
     * Saves the connection to a JSON object.
     * @param serializationObject the object to serialize to.
     */
    serialize(serializationObject = {}) {
        serializationObject.uniqueId = this.uniqueId;
        serializationObject.name = this.name;
        serializationObject._connectionType = this._connectionType;
        serializationObject.connectedPointIds = [];
        serializationObject.className = this.getClassName();
        for (const point of this._connectedPoint) {
            serializationObject.connectedPointIds.push(point.uniqueId);
        }
    }
    /**
     * @returns class name of the connection.
     */
    getClassName() {
        return "FGConnection";
    }
    /**
     * Deserialize from a object into this
     * @param serializationObject the object to deserialize from.
     */
    deserialize(serializationObject) {
        this.uniqueId = serializationObject.uniqueId;
        this.name = serializationObject.name;
        this._connectionType = serializationObject._connectionType;
        this.connectedPointIds = serializationObject.connectedPointIds;
    }
}

/**
 * Represents a connection point for data.
 * An unconnected input point can have a default value.
 * An output point will only have a value if it is connected to an input point. Furthermore,
 * if the point belongs to a "function" node, the node will run its function to update the value.
 */
class FlowGraphDataConnection extends FlowGraphConnection {
    /**
     * Create a new data connection point.
     * @param name the name of the connection
     * @param connectionType the type of the connection
     * @param ownerBlock the block that owns this connection
     * @param richType the type of the data in this block
     * @param _defaultValue the default value of the connection
     * @param _optional if the connection is optional
     */
    constructor(name, connectionType, ownerBlock, 
    /**
     * the type of the data in this block
     */
    richType, 
    /**
     * [any] the default value of the connection
     */
    _defaultValue = richType.defaultValue, 
    /**
     * [false] if the connection is optional
     */
    _optional = false) {
        super(name, connectionType, ownerBlock);
        this.richType = richType;
        this._defaultValue = _defaultValue;
        this._optional = _optional;
        this._isDisabled = false;
        /**
         * This is used for debugging purposes! It is the last value that was set to this connection with ANY context.
         * Do not use this value for anything else, as it might be wrong if used in a different context.
         */
        this._lastValue = null;
        /**
         * a data transformer function, if needed.
         * This can be used, for example, to force seconds into milliseconds output, if it makes sense in your case.
         */
        this.dataTransformer = null;
        /**
         * An observable that is triggered when the value of the connection changes.
         */
        this.onValueChangedObservable = new Observable();
    }
    /**
     * Whether or not the connection is optional.
     * Currently only used for UI control.
     */
    get optional() {
        return this._optional;
    }
    /**
     * is this connection disabled
     * If the connection is disabled you will not be able to connect anything to it.
     */
    get isDisabled() {
        return this._isDisabled;
    }
    set isDisabled(value) {
        if (this._isDisabled === value) {
            return;
        }
        this._isDisabled = value;
        if (this._isDisabled) {
            this.disconnectFromAll();
        }
    }
    /**
     * An output data block can connect to multiple input data blocks,
     * but an input data block can only connect to one output data block.
     * @returns true if the connection is singular
     */
    _isSingularConnection() {
        return this.connectionType === 0 /* FlowGraphConnectionType.Input */;
    }
    /**
     * Set the value of the connection in a specific context.
     * @param value the value to set
     * @param context the context to which the value is set
     */
    setValue(value, context) {
        // check if the value is different
        if (context._getConnectionValue(this) === value) {
            return;
        }
        context._setConnectionValue(this, value);
        this.onValueChangedObservable.notifyObservers(value);
    }
    /**
     * Reset the value of the connection to the default value.
     * @param context the context in which the value is reset
     */
    resetToDefaultValue(context) {
        context._setConnectionValue(this, this._defaultValue);
    }
    /**
     * Connect this point to another point.
     * @param point the point to connect to.
     */
    connectTo(point) {
        if (this._isDisabled) {
            return;
        }
        super.connectTo(point);
    }
    _getValueOrDefault(context) {
        const val = context._getConnectionValue(this) ?? this._defaultValue;
        return this.dataTransformer ? this.dataTransformer(val) : val;
    }
    /**
     * Gets the value of the connection in a specific context.
     * @param context the context from which the value is retrieved
     * @returns the value of the connection
     */
    getValue(context) {
        if (this.connectionType === 1 /* FlowGraphConnectionType.Output */) {
            context._notifyExecuteNode(this._ownerBlock);
            this._ownerBlock._updateOutputs(context);
            const value = this._getValueOrDefault(context);
            this._lastValue = value;
            return this.richType.typeTransformer ? this.richType.typeTransformer(value) : value;
        }
        const value = !this.isConnected() ? this._getValueOrDefault(context) : this._connectedPoint[0].getValue(context);
        this._lastValue = value;
        return this.richType.typeTransformer ? this.richType.typeTransformer(value) : value;
    }
    /**
     * @internal
     */
    _getLastValue() {
        return this._lastValue;
    }
    /**
     * @returns class name of the object.
     */
    getClassName() {
        return "FlowGraphDataConnection";
    }
    /**
     * Serializes this object.
     * @param serializationObject the object to serialize to
     */
    serialize(serializationObject = {}) {
        super.serialize(serializationObject);
        serializationObject.richType = {};
        this.richType.serialize(serializationObject.richType);
        serializationObject.optional = this._optional;
        defaultValueSerializationFunction("defaultValue", this._defaultValue, serializationObject);
    }
}
RegisterClass("FlowGraphDataConnection", FlowGraphDataConnection);

/**
 * A block in a flow graph. The most basic form
 * of a block has inputs and outputs that contain
 * data.
 */
class FlowGraphBlock {
    /** Constructor is protected so only subclasses can be instantiated
     * @param config optional configuration for this block
     * @internal - do not use directly. Extend this class instead.
     */
    constructor(
    /**
     * the configuration of the block
     */
    config) {
        this.config = config;
        /**
         * A randomly generated GUID for each block.
         */
        this.uniqueId = RandomGUID();
        this.name = this.config?.name ?? this.getClassName();
        this.dataInputs = [];
        this.dataOutputs = [];
    }
    /**
     * @internal
     * This function is called when the block needs to update its output flows.
     * @param _context the context in which it is running
     */
    _updateOutputs(_context) {
        // empty by default, overridden in data blocks
    }
    /**
     * Registers a data input on the block.
     * @param name the name of the input
     * @param richType the type of the input
     * @param defaultValue optional default value of the input. If not set, the rich type's default value will be used.
     * @returns the created connection
     */
    registerDataInput(name, richType, defaultValue) {
        const input = new FlowGraphDataConnection(name, 0 /* FlowGraphConnectionType.Input */, this, richType, defaultValue);
        this.dataInputs.push(input);
        return input;
    }
    /**
     * Registers a data output on the block.
     * @param name the name of the input
     * @param richType the type of the input
     * @param defaultValue optional default value of the input. If not set, the rich type's default value will be used.
     * @returns the created connection
     */
    registerDataOutput(name, richType, defaultValue) {
        const output = new FlowGraphDataConnection(name, 1 /* FlowGraphConnectionType.Output */, this, richType, defaultValue);
        this.dataOutputs.push(output);
        return output;
    }
    /**
     * Given the name of a data input, returns the connection if it exists
     * @param name the name of the input
     * @returns the connection if it exists, undefined otherwise
     */
    getDataInput(name) {
        return this.dataInputs.find((i) => i.name === name);
    }
    /**
     * Given the name of a data output, returns the connection if it exists
     * @param name the name of the output
     * @returns the connection if it exists, undefined otherwise
     */
    getDataOutput(name) {
        return this.dataOutputs.find((i) => i.name === name);
    }
    /**
     * Serializes this block
     * @param serializationObject the object to serialize to
     * @param _valueSerializeFunction a function that serializes a specific value
     */
    serialize(serializationObject = {}, _valueSerializeFunction = defaultValueSerializationFunction) {
        serializationObject.uniqueId = this.uniqueId;
        serializationObject.config = {};
        if (this.config) {
            const config = this.config;
            const keys = Object.keys(config);
            for (const key of keys) {
                _valueSerializeFunction(key, config[key], serializationObject.config);
            }
        }
        serializationObject.dataInputs = [];
        serializationObject.dataOutputs = [];
        serializationObject.className = this.getClassName();
        for (const input of this.dataInputs) {
            const serializedInput = {};
            input.serialize(serializedInput);
            serializationObject.dataInputs.push(serializedInput);
        }
        for (const output of this.dataOutputs) {
            const serializedOutput = {};
            output.serialize(serializedOutput);
            serializationObject.dataOutputs.push(serializedOutput);
        }
    }
    /**
     * Deserializes this block
     * @param _serializationObject the object to deserialize from
     */
    deserialize(_serializationObject) {
        // no-op by default
    }
    _log(context, action, payload) {
        context.logger?.addLogItem({
            action,
            payload,
            className: this.getClassName(),
            uniqueId: this.uniqueId,
        });
    }
    /**
     * Gets the class name of this block
     * @returns the class name
     */
    getClassName() {
        return "FlowGraphBlock";
    }
}

/**
 * Represents a connection point for a signal.
 * When an output point is activated, it will activate the connected input point.
 * When an input point is activated, it will execute the block it belongs to.
 */
class FlowGraphSignalConnection extends FlowGraphConnection {
    constructor() {
        super(...arguments);
        /**
         * The priority of the signal. Signals with higher priority will be executed first.
         * Set priority before adding the connection as sorting happens only when the connection is added.
         */
        this.priority = 0;
    }
    _isSingularConnection() {
        return false;
    }
    connectTo(point) {
        super.connectTo(point);
        // sort according to priority to handle execution order
        this._connectedPoint.sort((a, b) => b.priority - a.priority);
    }
    /**
     * @internal
     */
    _activateSignal(context) {
        context.logger?.addLogItem({
            action: "ActivateSignal" /* FlowGraphAction.ActivateSignal */,
            className: this._ownerBlock.getClassName(),
            uniqueId: this._ownerBlock.uniqueId,
            payload: {
                connectionType: this.connectionType,
                name: this.name,
            },
        });
        if (this.connectionType === 0 /* FlowGraphConnectionType.Input */) {
            context._notifyExecuteNode(this._ownerBlock);
            this._ownerBlock._execute(context, this);
            context._increaseExecutionId();
        }
        else {
            for (const connectedPoint of this._connectedPoint) {
                connectedPoint._activateSignal(context);
            }
        }
    }
}
RegisterClass("FlowGraphSignalConnection", FlowGraphSignalConnection);

/**
 * A block that executes some action. Always has an input signal (which is not used by event blocks).
 * Can have one or more output signals.
 */
class FlowGraphExecutionBlock extends FlowGraphBlock {
    constructor(config) {
        super(config);
        /**
         * The priority of the block. Higher priority blocks will be executed first.
         * Note that priority cannot be change AFTER the block was added as sorting happens when the block is added to the execution queue.
         */
        this.priority = 0;
        this.signalInputs = [];
        this.signalOutputs = [];
        this.in = this._registerSignalInput("in");
        this.error = this._registerSignalOutput("error");
    }
    _registerSignalInput(name) {
        const input = new FlowGraphSignalConnection(name, 0 /* FlowGraphConnectionType.Input */, this);
        this.signalInputs.push(input);
        return input;
    }
    _registerSignalOutput(name) {
        const output = new FlowGraphSignalConnection(name, 1 /* FlowGraphConnectionType.Output */, this);
        this.signalOutputs.push(output);
        return output;
    }
    _unregisterSignalInput(name) {
        const index = this.signalInputs.findIndex((input) => input.name === name);
        if (index !== -1) {
            this.signalInputs[index].dispose();
            this.signalInputs.splice(index, 1);
        }
    }
    _unregisterSignalOutput(name) {
        const index = this.signalOutputs.findIndex((output) => output.name === name);
        if (index !== -1) {
            this.signalOutputs[index].dispose();
            this.signalOutputs.splice(index, 1);
        }
    }
    _reportError(context, error) {
        this.error.payload = typeof error === "string" ? new Error(error) : error;
        this.error._activateSignal(context);
    }
    /**
     * Given a name of a signal input, return that input if it exists
     * @param name the name of the input
     * @returns if the input exists, the input. Otherwise, undefined.
     */
    getSignalInput(name) {
        return this.signalInputs.find((input) => input.name === name);
    }
    /**
     * Given a name of a signal output, return that input if it exists
     * @param name the name of the input
     * @returns if the input exists, the input. Otherwise, undefined.
     */
    getSignalOutput(name) {
        return this.signalOutputs.find((output) => output.name === name);
    }
    /**
     * Serializes this block
     * @param serializationObject the object to serialize in
     */
    serialize(serializationObject = {}) {
        super.serialize(serializationObject);
        serializationObject.signalInputs = [];
        serializationObject.signalOutputs = [];
        for (const input of this.signalInputs) {
            const serializedInput = {};
            input.serialize(serializedInput);
            serializationObject.signalInputs.push(serializedInput);
        }
        for (const output of this.signalOutputs) {
            const serializedOutput = {};
            output.serialize(serializedOutput);
            serializationObject.signalOutputs.push(serializedOutput);
        }
    }
    /**
     * Deserializes from an object
     * @param serializationObject the object to deserialize from
     */
    deserialize(serializationObject) {
        for (let i = 0; i < serializationObject.signalInputs.length; i++) {
            const signalInput = this.getSignalInput(serializationObject.signalInputs[i].name);
            if (signalInput) {
                signalInput.deserialize(serializationObject.signalInputs[i]);
            }
            else {
                throw new Error("Could not find signal input with name " + serializationObject.signalInputs[i].name + " in block " + serializationObject.className);
            }
        }
        for (let i = 0; i < serializationObject.signalOutputs.length; i++) {
            const signalOutput = this.getSignalOutput(serializationObject.signalOutputs[i].name);
            if (signalOutput) {
                signalOutput.deserialize(serializationObject.signalOutputs[i]);
            }
            else {
                throw new Error("Could not find signal output with name " + serializationObject.signalOutputs[i].name + " in block " + serializationObject.className);
            }
        }
    }
    /**
     * @returns the class name
     */
    getClassName() {
        return "FlowGraphExecutionBlock";
    }
}

/**
 * This class is responsible for coordinating the events that are triggered in the scene.
 * It registers all observers needed to track certain events and triggers the blocks that are listening to them.
 * Abstracting the events from the class will allow us to easily change the events that are being listened to, and trigger them in any order.
 */
class FlowGraphSceneEventCoordinator {
    constructor(scene) {
        /**
         * register to this observable to get flow graph event notifications.
         */
        this.onEventTriggeredObservable = new Observable();
        /**
         * Was scene-ready already triggered?
         */
        this.sceneReadyTriggered = false;
        this._pointerUnderMeshState = {};
        this._startingTime = 0;
        this._scene = scene;
        this._initialize();
    }
    _initialize() {
        this._sceneReadyObserver = this._scene.onReadyObservable.add(() => {
            if (!this.sceneReadyTriggered) {
                this.onEventTriggeredObservable.notifyObservers({ type: "SceneReady" /* FlowGraphEventType.SceneReady */ });
                this.sceneReadyTriggered = true;
            }
        });
        this._sceneDisposeObserver = this._scene.onDisposeObservable.add(() => {
            this.onEventTriggeredObservable.notifyObservers({ type: "SceneDispose" /* FlowGraphEventType.SceneDispose */ });
        });
        this._sceneOnBeforeRenderObserver = this._scene.onBeforeRenderObservable.add(() => {
            const deltaTime = this._scene.getEngine().getDeltaTime() / 1000; // set in seconds
            this.onEventTriggeredObservable.notifyObservers({
                type: "SceneBeforeRender" /* FlowGraphEventType.SceneBeforeRender */,
                payload: {
                    timeSinceStart: this._startingTime,
                    deltaTime,
                },
            });
            this._startingTime += deltaTime;
        });
        this._meshPickedObserver = this._scene.onPointerObservable.add((pointerInfo) => {
            this.onEventTriggeredObservable.notifyObservers({ type: "MeshPick" /* FlowGraphEventType.MeshPick */, payload: pointerInfo });
        }, PointerEventTypes.POINTERPICK); // should it be pointerdown?
        this._meshUnderPointerObserver = this._scene.onMeshUnderPointerUpdatedObservable.add((data) => {
            // check if the data has changed. Check the state of the last change and see if it is a mesh or null.
            // if it is a mesh and the previous state was null, trigger over event. If it is null and the previous state was a mesh, trigger out event.
            // if it is a mesh and the previous state was a mesh, trigger out from the old mesh and over the new mesh
            // if it is null and the previous state was null, do nothing.
            const pointerId = data.pointerId;
            const mesh = data.mesh;
            const previousState = this._pointerUnderMeshState[pointerId];
            if (!previousState && mesh) {
                this.onEventTriggeredObservable.notifyObservers({ type: "PointerOver" /* FlowGraphEventType.PointerOver */, payload: { pointerId, mesh } });
            }
            else if (previousState && !mesh) {
                this.onEventTriggeredObservable.notifyObservers({ type: "PointerOut" /* FlowGraphEventType.PointerOut */, payload: { pointerId, mesh: previousState } });
            }
            else if (previousState && mesh && previousState !== mesh) {
                this.onEventTriggeredObservable.notifyObservers({ type: "PointerOut" /* FlowGraphEventType.PointerOut */, payload: { pointerId, mesh: previousState, over: mesh } });
                this.onEventTriggeredObservable.notifyObservers({ type: "PointerOver" /* FlowGraphEventType.PointerOver */, payload: { pointerId, mesh, out: previousState } });
            }
            this._pointerUnderMeshState[pointerId] = mesh;
        }, PointerEventTypes.POINTERMOVE);
    }
    dispose() {
        this._sceneDisposeObserver?.remove();
        this._sceneReadyObserver?.remove();
        this._sceneOnBeforeRenderObserver?.remove();
        this._meshPickedObserver?.remove();
        this._meshUnderPointerObserver?.remove();
        this.onEventTriggeredObservable.clear();
    }
}

/**
 * @internal
 * Returns if mesh1 is a descendant of mesh2
 * @param mesh1
 * @param mesh2
 * @returns
 */
function _IsDescendantOf(mesh1, mesh2) {
    return !!(mesh1.parent && (mesh1.parent === mesh2 || _IsDescendantOf(mesh1.parent, mesh2)));
}
/**
 * @internal
 */
function _GetClassNameOf(v) {
    if (v.getClassName) {
        return v.getClassName();
    }
    return;
}
/**
 * @internal
 * Check if two classname are the same and are vector or quaternion classes.
 * @param className the first class name
 * @param className2 the second class name
 * @returns whether the two class names are the same and are vector or quaternion classes.
 */
function _AreSameVectorOrQuaternionClass(className, className2) {
    return (className === className2 &&
        (className === "Vector2" /* FlowGraphTypes.Vector2 */ || className === "Vector3" /* FlowGraphTypes.Vector3 */ || className === "Vector4" /* FlowGraphTypes.Vector4 */ || className === "Quaternion" /* FlowGraphTypes.Quaternion */));
}
/**
 * @internal
 * Check if two classname are the same and are matrix classes.
 * @param className the first class name
 * @param className2 the second class name
 * @returns whether the two class names are the same and are matrix classes.
 */
function _AreSameMatrixClass(className, className2) {
    return className === className2 && (className === "Matrix" /* FlowGraphTypes.Matrix */ || className === "Matrix2D" /* FlowGraphTypes.Matrix2D */ || className === "Matrix3D" /* FlowGraphTypes.Matrix3D */);
}
/**
 * @internal
 * Check if two classname are the same and are integer classes.
 * @param className the first class name
 * @param className2 the second class name
 * @returns whether the two class names are the same and are integer classes.
 */
function _AreSameIntegerClass(className, className2) {
    return className === "FlowGraphInteger" && className2 === "FlowGraphInteger";
}
/**
 * Check if an object has a numeric value.
 * @param a the object to check if it is a number.
 * @param validIfNaN whether to consider NaN as a valid number.
 * @returns whether a is a FlowGraphNumber (Integer or number).
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
function isNumeric(a, validIfNaN) {
    const isNumeric = typeof a === "number" || typeof a?.value === "number";
    if (isNumeric && !validIfNaN) {
        return !isNaN(getNumericValue(a));
    }
    return isNumeric;
}
/**
 * Get the numeric value of a FlowGraphNumber.
 * @param a the object to get the numeric value from.
 * @returns the numeric value.
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
function getNumericValue(a) {
    return typeof a === "number" ? a : a.value;
}

var FlowGraphState;
(function (FlowGraphState) {
    /**
     * The graph is stopped
     */
    FlowGraphState[FlowGraphState["Stopped"] = 0] = "Stopped";
    /**
     * The graph is running
     */
    FlowGraphState[FlowGraphState["Started"] = 1] = "Started";
})(FlowGraphState || (FlowGraphState = {}));
/**
 * Class used to represent a flow graph.
 * A flow graph is a graph of blocks that can be used to create complex logic.
 * Blocks can be added to the graph and connected to each other.
 * The graph can then be started, which will init and start all of its event blocks.
 *
 * @experimental FlowGraph is still in development and is subject to change.
 */
class FlowGraph {
    /**
     * The state of the graph
     */
    get state() {
        return this._state;
    }
    /**
     * The state of the graph
     */
    set state(value) {
        this._state = value;
        this.onStateChangedObservable.notifyObservers(value);
    }
    /**
     * Construct a Flow Graph
     * @param params construction parameters. currently only the scene
     */
    constructor(params) {
        /**
         * An observable that is triggered when the state of the graph changes.
         */
        this.onStateChangedObservable = new Observable();
        /** @internal */
        this._eventBlocks = {
            ["SceneReady" /* FlowGraphEventType.SceneReady */]: [],
            ["SceneDispose" /* FlowGraphEventType.SceneDispose */]: [],
            ["SceneBeforeRender" /* FlowGraphEventType.SceneBeforeRender */]: [],
            ["MeshPick" /* FlowGraphEventType.MeshPick */]: [],
            ["PointerDown" /* FlowGraphEventType.PointerDown */]: [],
            ["PointerUp" /* FlowGraphEventType.PointerUp */]: [],
            ["PointerMove" /* FlowGraphEventType.PointerMove */]: [],
            ["PointerOver" /* FlowGraphEventType.PointerOver */]: [],
            ["PointerOut" /* FlowGraphEventType.PointerOut */]: [],
            ["SceneAfterRender" /* FlowGraphEventType.SceneAfterRender */]: [],
            ["NoTrigger" /* FlowGraphEventType.NoTrigger */]: [],
        };
        this._executionContexts = [];
        /**
         * The state of the graph
         */
        this._state = 0 /* FlowGraphState.Stopped */;
        this._scene = params.scene;
        this._sceneEventCoordinator = new FlowGraphSceneEventCoordinator(this._scene);
        this._coordinator = params.coordinator;
        this._eventObserver = this._sceneEventCoordinator.onEventTriggeredObservable.add((event) => {
            for (const context of this._executionContexts) {
                const order = this._getContextualOrder(event.type, context);
                for (const block of order) {
                    // iterate contexts
                    if (!block._executeEvent(context, event.payload)) {
                        break;
                    }
                }
            }
            // custom behavior(s) of specific events
            switch (event.type) {
                case "SceneReady" /* FlowGraphEventType.SceneReady */:
                    this._sceneEventCoordinator.sceneReadyTriggered = true;
                    break;
                case "SceneBeforeRender" /* FlowGraphEventType.SceneBeforeRender */:
                    for (const context of this._executionContexts) {
                        context._notifyOnTick(event.payload);
                    }
                    break;
                case "SceneDispose" /* FlowGraphEventType.SceneDispose */:
                    this.dispose();
                    break;
            }
        });
    }
    /**
     * Create a context. A context represents one self contained execution for the graph, with its own variables.
     * @returns the context, where you can get and set variables
     */
    createContext() {
        const context = new FlowGraphContext({ scene: this._scene, coordinator: this._coordinator });
        this._executionContexts.push(context);
        return context;
    }
    /**
     * Returns the execution context at a given index
     * @param index the index of the context
     * @returns the execution context at that index
     */
    getContext(index) {
        return this._executionContexts[index];
    }
    /**
     * Add an event block. When the graph is started, it will start listening to events
     * from the block and execute the graph when they are triggered.
     * @param block the event block to be added
     */
    addEventBlock(block) {
        if (block.type === "PointerOver" /* FlowGraphEventType.PointerOver */ || block.type === "PointerOut" /* FlowGraphEventType.PointerOut */) {
            this._scene.constantlyUpdateMeshUnderPointer = true;
        }
        // don't add if NoTrigger, but still start the pending tasks
        if (block.type !== "NoTrigger" /* FlowGraphEventType.NoTrigger */) {
            this._eventBlocks[block.type].push(block);
        }
        // if already started, sort and add to the pending
        if (this.state === 1 /* FlowGraphState.Started */) {
            for (const context of this._executionContexts) {
                block._startPendingTasks(context);
            }
        }
        else {
            this.onStateChangedObservable.addOnce((state) => {
                if (state === 1 /* FlowGraphState.Started */) {
                    for (const context of this._executionContexts) {
                        block._startPendingTasks(context);
                    }
                }
            });
        }
    }
    /**
     * Starts the flow graph. Initializes the event blocks and starts listening to events.
     */
    start() {
        if (this.state === 1 /* FlowGraphState.Started */) {
            return;
        }
        if (this._executionContexts.length === 0) {
            this.createContext();
        }
        this.onStateChangedObservable.add((state) => {
            if (state === 1 /* FlowGraphState.Started */) {
                this._startPendingEvents();
                // the only event we need to check is the scene ready event. If the scene is already ready when the graph starts, we should start the pending tasks.
                if (this._scene.isReady(true)) {
                    this._sceneEventCoordinator.onEventTriggeredObservable.notifyObservers({ type: "SceneReady" /* FlowGraphEventType.SceneReady */ });
                }
            }
        });
        this.state = 1 /* FlowGraphState.Started */;
    }
    _startPendingEvents() {
        for (const context of this._executionContexts) {
            for (const type in this._eventBlocks) {
                const order = this._getContextualOrder(type, context);
                for (const block of order) {
                    block._startPendingTasks(context);
                }
            }
        }
    }
    _getContextualOrder(type, context) {
        const order = this._eventBlocks[type].sort((a, b) => b.initPriority - a.initPriority);
        if (type === "MeshPick" /* FlowGraphEventType.MeshPick */) {
            const meshPickOrder = [];
            for (const block1 of order) {
                // If the block is a mesh pick, guarantee that picks of children meshes come before picks of parent meshes
                const mesh1 = block1.asset.getValue(context);
                let i = 0;
                for (; i < order.length; i++) {
                    const block2 = order[i];
                    const mesh2 = block2.asset.getValue(context);
                    if (mesh1 && mesh2 && _IsDescendantOf(mesh1, mesh2)) {
                        break;
                    }
                }
                meshPickOrder.splice(i, 0, block1);
            }
            return meshPickOrder;
        }
        return order;
    }
    /**
     * Disposes of the flow graph. Cancels any pending tasks and removes all event listeners.
     */
    dispose() {
        if (this.state === 0 /* FlowGraphState.Stopped */) {
            return;
        }
        this.state = 0 /* FlowGraphState.Stopped */;
        for (const context of this._executionContexts) {
            context._clearPendingBlocks();
        }
        this._executionContexts.length = 0;
        for (const type in this._eventBlocks) {
            this._eventBlocks[type].length = 0;
        }
        this._eventObserver?.remove();
        this._sceneEventCoordinator.dispose();
    }
    /**
     * Executes a function in all blocks of a flow graph, starting with the event blocks.
     * @param visitor the function to execute.
     */
    visitAllBlocks(visitor) {
        const visitList = [];
        const idsAddedToVisitList = new Set();
        for (const type in this._eventBlocks) {
            for (const block of this._eventBlocks[type]) {
                visitList.push(block);
                idsAddedToVisitList.add(block.uniqueId);
            }
        }
        while (visitList.length > 0) {
            const block = visitList.pop();
            visitor(block);
            for (const dataIn of block.dataInputs) {
                for (const connection of dataIn._connectedPoint) {
                    if (!idsAddedToVisitList.has(connection._ownerBlock.uniqueId)) {
                        visitList.push(connection._ownerBlock);
                        idsAddedToVisitList.add(connection._ownerBlock.uniqueId);
                    }
                }
            }
            if (block instanceof FlowGraphExecutionBlock) {
                for (const signalOut of block.signalOutputs) {
                    for (const connection of signalOut._connectedPoint) {
                        if (!idsAddedToVisitList.has(connection._ownerBlock.uniqueId)) {
                            visitList.push(connection._ownerBlock);
                            idsAddedToVisitList.add(connection._ownerBlock.uniqueId);
                        }
                    }
                }
            }
        }
    }
    /**
     * Serializes a graph
     * @param serializationObject the object to write the values in
     * @param valueSerializeFunction a function to serialize complex values
     */
    serialize(serializationObject = {}, valueSerializeFunction) {
        serializationObject.allBlocks = [];
        this.visitAllBlocks((block) => {
            const serializedBlock = {};
            block.serialize(serializedBlock);
            serializationObject.allBlocks.push(serializedBlock);
        });
        serializationObject.executionContexts = [];
        for (const context of this._executionContexts) {
            const serializedContext = {};
            context.serialize(serializedContext, valueSerializeFunction);
            serializationObject.executionContexts.push(serializedContext);
        }
    }
}

/**
 * This class holds all of the existing flow graphs and is responsible for creating new ones.
 * It also handles starting/stopping multiple graphs and communication between them through an Event Coordinator
 * This is the entry point for the flow graph system.
 * @experimental This class is still in development and is subject to change.
 */
class FlowGraphCoordinator {
    constructor(
    /**
     * the configuration of the block
     */
    config) {
        this.config = config;
        /**
         * When set to true (default) custom events will be dispatched synchronously.
         * This means that the events will be dispatched immediately when they are triggered.
         */
        this.dispatchEventsSynchronously = true;
        this._flowGraphs = [];
        this._customEventsMap = new Map();
        this._eventExecutionCounter = new Map();
        this._executeOnNextFrame = [];
        this._eventUniqueId = 0;
        // When the scene is disposed, dispose all graphs currently running on it.
        this._disposeObserver = this.config.scene.onDisposeObservable.add(() => {
            this.dispose();
        });
        this._onBeforeRenderObserver = this.config.scene.onBeforeRenderObservable.add(() => {
            // Reset the event execution counter at the beginning of each frame.
            this._eventExecutionCounter.clear();
            // duplicate the _executeOnNextFrame array to avoid modifying it while iterating over it
            const executeOnNextFrame = this._executeOnNextFrame.slice(0);
            if (executeOnNextFrame.length) {
                // Execute the events that were triggered on the next frame.
                for (const event of executeOnNextFrame) {
                    this.notifyCustomEvent(event.id, event.data, false);
                    // remove the event from the array
                    const index = this._executeOnNextFrame.findIndex((e) => e.uniqueId === event.uniqueId);
                    if (index !== -1) {
                        this._executeOnNextFrame.splice(index, 1);
                    }
                }
            }
        });
        // Add itself to the SceneCoordinators list for the Inspector.
        const coordinators = FlowGraphCoordinator.SceneCoordinators.get(this.config.scene) ?? [];
        coordinators.push(this);
    }
    /**
     * Creates a new flow graph and adds it to the list of existing flow graphs
     * @returns a new flow graph
     */
    createGraph() {
        const graph = new FlowGraph({ scene: this.config.scene, coordinator: this });
        this._flowGraphs.push(graph);
        return graph;
    }
    /**
     * Removes a flow graph from the list of existing flow graphs and disposes it
     * @param graph the graph to remove
     */
    removeGraph(graph) {
        const index = this._flowGraphs.indexOf(graph);
        if (index !== -1) {
            graph.dispose();
            this._flowGraphs.splice(index, 1);
        }
    }
    /**
     * Starts all graphs
     */
    start() {
        for (const graph of this._flowGraphs) {
            graph.start();
        }
    }
    /**
     * Disposes all graphs
     */
    dispose() {
        for (const graph of this._flowGraphs) {
            graph.dispose();
        }
        this._flowGraphs.length = 0;
        this._disposeObserver?.remove();
        this._onBeforeRenderObserver?.remove();
        // Remove itself from the SceneCoordinators list for the Inspector.
        const coordinators = FlowGraphCoordinator.SceneCoordinators.get(this.config.scene) ?? [];
        const index = coordinators.indexOf(this);
        if (index !== -1) {
            coordinators.splice(index, 1);
        }
    }
    /**
     * Serializes this coordinator to a JSON object.
     * @param serializationObject the object to serialize to
     * @param valueSerializeFunction the function to use to serialize the value
     */
    serialize(serializationObject, valueSerializeFunction) {
        serializationObject._flowGraphs = [];
        for (const graph of this._flowGraphs) {
            const serializedGraph = {};
            graph.serialize(serializedGraph, valueSerializeFunction);
            serializationObject._flowGraphs.push(serializedGraph);
        }
        serializationObject.dispatchEventsSynchronously = this.dispatchEventsSynchronously;
    }
    /**
     * Gets the list of flow graphs
     */
    get flowGraphs() {
        return this._flowGraphs;
    }
    /**
     * Get an observable that will be notified when the event with the given id is fired.
     * @param id the id of the event
     * @returns the observable for the event
     */
    getCustomEventObservable(id) {
        let observable = this._customEventsMap.get(id);
        if (!observable) {
            // receive event is initialized before scene start, so no need to notify if triggered. but possible!
            observable = new Observable( /*undefined, true*/);
            this._customEventsMap.set(id, observable);
        }
        return observable;
    }
    /**
     * Notifies the observable for the given event id with the given data.
     * @param id the id of the event
     * @param data the data to send with the event
     * @param async if true, the event will be dispatched asynchronously
     */
    notifyCustomEvent(id, data, async = !this.dispatchEventsSynchronously) {
        if (async) {
            this._executeOnNextFrame.push({ id, data, uniqueId: this._eventUniqueId++ });
            return;
        }
        // check if we are not exceeding the max number of events
        if (this._eventExecutionCounter.has(id)) {
            const count = this._eventExecutionCounter.get(id);
            this._eventExecutionCounter.set(id, count + 1);
            if (count >= FlowGraphCoordinator.MaxEventTypeExecutionPerFrame) {
                if (count === FlowGraphCoordinator.MaxEventTypeExecutionPerFrame) {
                    Logger.Warn(`FlowGraphCoordinator: Too many executions of event "${id}".`);
                }
                return;
            }
        }
        else {
            this._eventExecutionCounter.set(id, 1);
        }
        const observable = this._customEventsMap.get(id);
        if (observable) {
            observable.notifyObservers(data);
        }
    }
}
/**
 * The maximum number of events per type.
 * This is used to limit the number of events that can be created in a single scene.
 * This is to prevent infinite loops.
 */
FlowGraphCoordinator.MaxEventsPerType = 30;
/**
 * The maximum number of execution of a specific event in a single frame.
 */
FlowGraphCoordinator.MaxEventTypeExecutionPerFrame = 30;
/**
 * @internal
 * A list of all the coordinators per scene. Will be used by the inspector
 */
FlowGraphCoordinator.SceneCoordinators = new Map();

const CustomBlocks = {};
/**
 * If you want to add a new block to the block factory, you should use this function.
 * Please be sure to choose a unique name and define the responsible module.
 * @param module the name of the module that is responsible for the block
 * @param blockName the name of the block. This should be unique.
 * @param factory an async factory function to generate the block
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
function addToBlockFactory(module, blockName, factory) {
    CustomBlocks[`${module}/${blockName}`] = factory;
}
/**
 * a function to get a factory function for a block.
 * @param blockName the block name to initialize. If the block comes from an external module, the name should be in the format "module/blockName"
 * @returns an async factory function that will return the block class when called.
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
function blockFactory(blockName) {
    switch (blockName) {
        case "FlowGraphPlayAnimationBlock" /* FlowGraphBlockNames.PlayAnimation */:
            return async () => (await __vitePreload(async () => { const {FlowGraphPlayAnimationBlock} = await import('./flowGraphPlayAnimationBlock-BkKzjOFT.js');return { FlowGraphPlayAnimationBlock }},true              ?__vite__mapDeps([0,1,2,3,4,5,6,7]):void 0,import.meta.url)).FlowGraphPlayAnimationBlock;
        case "FlowGraphStopAnimationBlock" /* FlowGraphBlockNames.StopAnimation */:
            return async () => (await __vitePreload(async () => { const {FlowGraphStopAnimationBlock} = await import('./flowGraphStopAnimationBlock-Cf1LB0Bk.js');return { FlowGraphStopAnimationBlock }},true              ?__vite__mapDeps([8,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphStopAnimationBlock;
        case "FlowGraphPauseAnimationBlock" /* FlowGraphBlockNames.PauseAnimation */:
            return async () => (await __vitePreload(async () => { const {FlowGraphPauseAnimationBlock} = await import('./flowGraphPauseAnimationBlock-B9i08t1Q.js');return { FlowGraphPauseAnimationBlock }},true              ?__vite__mapDeps([9,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphPauseAnimationBlock;
        case "FlowGraphInterpolationBlock" /* FlowGraphBlockNames.ValueInterpolation */:
            return async () => (await __vitePreload(async () => { const {FlowGraphInterpolationBlock} = await import('./flowGraphInterpolationBlock-a-o18QBY.js');return { FlowGraphInterpolationBlock }},true              ?__vite__mapDeps([10,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphInterpolationBlock;
        case "FlowGraphSceneReadyEventBlock" /* FlowGraphBlockNames.SceneReadyEvent */:
            return async () => (await __vitePreload(async () => { const {FlowGraphSceneReadyEventBlock} = await import('./flowGraphSceneReadyEventBlock-vtf4yMOr.js');return { FlowGraphSceneReadyEventBlock }},true              ?__vite__mapDeps([11,2,3,4,1,7]):void 0,import.meta.url)).FlowGraphSceneReadyEventBlock;
        case "FlowGraphSceneTickEventBlock" /* FlowGraphBlockNames.SceneTickEvent */:
            return async () => (await __vitePreload(async () => { const {FlowGraphSceneTickEventBlock} = await import('./flowGraphSceneTickEventBlock-DRNAn94I.js');return { FlowGraphSceneTickEventBlock }},true              ?__vite__mapDeps([12,2,3,4,1,7]):void 0,import.meta.url)).FlowGraphSceneTickEventBlock;
        case "FlowGraphSendCustomEventBlock" /* FlowGraphBlockNames.SendCustomEvent */:
            return async () => (await __vitePreload(async () => { const {FlowGraphSendCustomEventBlock} = await import('./flowGraphSendCustomEventBlock-6rVFn3mr.js');return { FlowGraphSendCustomEventBlock }},true              ?__vite__mapDeps([13,2,3,4,1,7]):void 0,import.meta.url)).FlowGraphSendCustomEventBlock;
        case "FlowGraphReceiveCustomEventBlock" /* FlowGraphBlockNames.ReceiveCustomEvent */:
            return async () => (await __vitePreload(async () => { const {FlowGraphReceiveCustomEventBlock} = await import('./flowGraphReceiveCustomEventBlock-CC4v1Q-T.js');return { FlowGraphReceiveCustomEventBlock }},true              ?__vite__mapDeps([14,2,3,4,1,7]):void 0,import.meta.url)).FlowGraphReceiveCustomEventBlock;
        case "FlowGraphMeshPickEventBlock" /* FlowGraphBlockNames.MeshPickEvent */:
            return async () => (await __vitePreload(async () => { const {FlowGraphMeshPickEventBlock} = await import('./flowGraphMeshPickEventBlock-BaFqgJHG.js');return { FlowGraphMeshPickEventBlock }},true              ?__vite__mapDeps([15,2,3,4,1,7]):void 0,import.meta.url)).FlowGraphMeshPickEventBlock;
        case "FlowGraphEBlock" /* FlowGraphBlockNames.E */:
            return async () => (await __vitePreload(async () => { const {FlowGraphEBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphEBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphEBlock;
        case "FlowGraphPIBlock" /* FlowGraphBlockNames.PI */:
            return async () => (await __vitePreload(async () => { const {FlowGraphPiBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphPiBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphPiBlock;
        case "FlowGraphInfBlock" /* FlowGraphBlockNames.Inf */:
            return async () => (await __vitePreload(async () => { const {FlowGraphInfBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphInfBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphInfBlock;
        case "FlowGraphNaNBlock" /* FlowGraphBlockNames.NaN */:
            return async () => (await __vitePreload(async () => { const {FlowGraphNaNBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphNaNBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphNaNBlock;
        case "FlowGraphRandomBlock" /* FlowGraphBlockNames.Random */:
            return async () => (await __vitePreload(async () => { const {FlowGraphRandomBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphRandomBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphRandomBlock;
        case "FlowGraphAddBlock" /* FlowGraphBlockNames.Add */:
            return async () => (await __vitePreload(async () => { const {FlowGraphAddBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphAddBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphAddBlock;
        case "FlowGraphSubtractBlock" /* FlowGraphBlockNames.Subtract */:
            return async () => (await __vitePreload(async () => { const {FlowGraphSubtractBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphSubtractBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphSubtractBlock;
        case "FlowGraphMultiplyBlock" /* FlowGraphBlockNames.Multiply */:
            return async () => (await __vitePreload(async () => { const {FlowGraphMultiplyBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphMultiplyBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphMultiplyBlock;
        case "FlowGraphDivideBlock" /* FlowGraphBlockNames.Divide */:
            return async () => (await __vitePreload(async () => { const {FlowGraphDivideBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphDivideBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphDivideBlock;
        case "FlowGraphAbsBlock" /* FlowGraphBlockNames.Abs */:
            return async () => (await __vitePreload(async () => { const {FlowGraphAbsBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphAbsBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphAbsBlock;
        case "FlowGraphSignBlock" /* FlowGraphBlockNames.Sign */:
            return async () => (await __vitePreload(async () => { const {FlowGraphSignBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphSignBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphSignBlock;
        case "FlowGraphTruncBlock" /* FlowGraphBlockNames.Trunc */:
            return async () => (await __vitePreload(async () => { const {FlowGraphTruncBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphTruncBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphTruncBlock;
        case "FlowGraphFloorBlock" /* FlowGraphBlockNames.Floor */:
            return async () => (await __vitePreload(async () => { const {FlowGraphFloorBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphFloorBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphFloorBlock;
        case "FlowGraphCeilBlock" /* FlowGraphBlockNames.Ceil */:
            return async () => (await __vitePreload(async () => { const {FlowGraphCeilBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphCeilBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphCeilBlock;
        case "FlowGraphRoundBlock" /* FlowGraphBlockNames.Round */:
            return async () => (await __vitePreload(async () => { const {FlowGraphRoundBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphRoundBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphRoundBlock;
        case "FlowGraphFractBlock" /* FlowGraphBlockNames.Fraction */:
            return async () => (await __vitePreload(async () => { const {FlowGraphFractionBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphFractionBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphFractionBlock;
        case "FlowGraphNegationBlock" /* FlowGraphBlockNames.Negation */:
            return async () => (await __vitePreload(async () => { const {FlowGraphNegationBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphNegationBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphNegationBlock;
        case "FlowGraphModuloBlock" /* FlowGraphBlockNames.Modulo */:
            return async () => (await __vitePreload(async () => { const {FlowGraphModuloBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphModuloBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphModuloBlock;
        case "FlowGraphMinBlock" /* FlowGraphBlockNames.Min */:
            return async () => (await __vitePreload(async () => { const {FlowGraphMinBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphMinBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphMinBlock;
        case "FlowGraphMaxBlock" /* FlowGraphBlockNames.Max */:
            return async () => (await __vitePreload(async () => { const {FlowGraphMaxBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphMaxBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphMaxBlock;
        case "FlowGraphClampBlock" /* FlowGraphBlockNames.Clamp */:
            return async () => (await __vitePreload(async () => { const {FlowGraphClampBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphClampBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphClampBlock;
        case "FlowGraphSaturateBlock" /* FlowGraphBlockNames.Saturate */:
            return async () => (await __vitePreload(async () => { const {FlowGraphSaturateBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphSaturateBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphSaturateBlock;
        case "FlowGraphMathInterpolationBlock" /* FlowGraphBlockNames.MathInterpolation */:
            return async () => (await __vitePreload(async () => { const {FlowGraphMathInterpolationBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphMathInterpolationBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphMathInterpolationBlock;
        case "FlowGraphEqualityBlock" /* FlowGraphBlockNames.Equality */:
            return async () => (await __vitePreload(async () => { const {FlowGraphEqualityBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphEqualityBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphEqualityBlock;
        case "FlowGraphLessThanBlock" /* FlowGraphBlockNames.LessThan */:
            return async () => (await __vitePreload(async () => { const {FlowGraphLessThanBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphLessThanBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphLessThanBlock;
        case "FlowGraphLessThanOrEqualBlock" /* FlowGraphBlockNames.LessThanOrEqual */:
            return async () => (await __vitePreload(async () => { const {FlowGraphLessThanOrEqualBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphLessThanOrEqualBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphLessThanOrEqualBlock;
        case "FlowGraphGreaterThanBlock" /* FlowGraphBlockNames.GreaterThan */:
            return async () => (await __vitePreload(async () => { const {FlowGraphGreaterThanBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphGreaterThanBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphGreaterThanBlock;
        case "FlowGraphGreaterThanOrEqualBlock" /* FlowGraphBlockNames.GreaterThanOrEqual */:
            return async () => (await __vitePreload(async () => { const {FlowGraphGreaterThanOrEqualBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphGreaterThanOrEqualBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphGreaterThanOrEqualBlock;
        case "FlowGraphIsNaNBlock" /* FlowGraphBlockNames.IsNaN */:
            return async () => (await __vitePreload(async () => { const {FlowGraphIsNanBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphIsNanBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphIsNanBlock;
        case "FlowGraphIsInfBlock" /* FlowGraphBlockNames.IsInfinity */:
            return async () => (await __vitePreload(async () => { const {FlowGraphIsInfinityBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphIsInfinityBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphIsInfinityBlock;
        case "FlowGraphDegToRadBlock" /* FlowGraphBlockNames.DegToRad */:
            return async () => (await __vitePreload(async () => { const {FlowGraphDegToRadBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphDegToRadBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphDegToRadBlock;
        case "FlowGraphRadToDegBlock" /* FlowGraphBlockNames.RadToDeg */:
            return async () => (await __vitePreload(async () => { const {FlowGraphRadToDegBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphRadToDegBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphRadToDegBlock;
        case "FlowGraphSinBlock" /* FlowGraphBlockNames.Sin */:
            return async () => (await __vitePreload(async () => { const {FlowGraphSinBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphSinBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphSinBlock;
        case "FlowGraphCosBlock" /* FlowGraphBlockNames.Cos */:
            return async () => (await __vitePreload(async () => { const {FlowGraphCosBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphCosBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphCosBlock;
        case "FlowGraphTanBlock" /* FlowGraphBlockNames.Tan */:
            return async () => (await __vitePreload(async () => { const {FlowGraphTanBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphTanBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphTanBlock;
        case "FlowGraphASinBlock" /* FlowGraphBlockNames.Asin */:
            return async () => (await __vitePreload(async () => { const {FlowGraphAsinBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphAsinBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphAsinBlock;
        case "FlowGraphACosBlock" /* FlowGraphBlockNames.Acos */:
            return async () => (await __vitePreload(async () => { const {FlowGraphAcosBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphAcosBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphAcosBlock;
        case "FlowGraphATanBlock" /* FlowGraphBlockNames.Atan */:
            return async () => (await __vitePreload(async () => { const {FlowGraphAtanBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphAtanBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphAtanBlock;
        case "FlowGraphATan2Block" /* FlowGraphBlockNames.Atan2 */:
            return async () => (await __vitePreload(async () => { const {FlowGraphAtan2Block} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphAtan2Block }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphAtan2Block;
        case "FlowGraphSinhBlock" /* FlowGraphBlockNames.Sinh */:
            return async () => (await __vitePreload(async () => { const {FlowGraphSinhBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphSinhBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphSinhBlock;
        case "FlowGraphCoshBlock" /* FlowGraphBlockNames.Cosh */:
            return async () => (await __vitePreload(async () => { const {FlowGraphCoshBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphCoshBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphCoshBlock;
        case "FlowGraphTanhBlock" /* FlowGraphBlockNames.Tanh */:
            return async () => (await __vitePreload(async () => { const {FlowGraphTanhBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphTanhBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphTanhBlock;
        case "FlowGraphASinhBlock" /* FlowGraphBlockNames.Asinh */:
            return async () => (await __vitePreload(async () => { const {FlowGraphAsinhBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphAsinhBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphAsinhBlock;
        case "FlowGraphACoshBlock" /* FlowGraphBlockNames.Acosh */:
            return async () => (await __vitePreload(async () => { const {FlowGraphAcoshBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphAcoshBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphAcoshBlock;
        case "FlowGraphATanhBlock" /* FlowGraphBlockNames.Atanh */:
            return async () => (await __vitePreload(async () => { const {FlowGraphAtanhBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphAtanhBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphAtanhBlock;
        case "FlowGraphExponentialBlock" /* FlowGraphBlockNames.Exponential */:
            return async () => (await __vitePreload(async () => { const {FlowGraphExpBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphExpBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphExpBlock;
        case "FlowGraphLogBlock" /* FlowGraphBlockNames.Log */:
            return async () => (await __vitePreload(async () => { const {FlowGraphLogBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphLogBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphLogBlock;
        case "FlowGraphLog2Block" /* FlowGraphBlockNames.Log2 */:
            return async () => (await __vitePreload(async () => { const {FlowGraphLog2Block} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphLog2Block }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphLog2Block;
        case "FlowGraphLog10Block" /* FlowGraphBlockNames.Log10 */:
            return async () => (await __vitePreload(async () => { const {FlowGraphLog10Block} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphLog10Block }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphLog10Block;
        case "FlowGraphSquareRootBlock" /* FlowGraphBlockNames.SquareRoot */:
            return async () => (await __vitePreload(async () => { const {FlowGraphSquareRootBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphSquareRootBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphSquareRootBlock;
        case "FlowGraphPowerBlock" /* FlowGraphBlockNames.Power */:
            return async () => (await __vitePreload(async () => { const {FlowGraphPowerBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphPowerBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphPowerBlock;
        case "FlowGraphCubeRootBlock" /* FlowGraphBlockNames.CubeRoot */:
            return async () => (await __vitePreload(async () => { const {FlowGraphCubeRootBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphCubeRootBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphCubeRootBlock;
        case "FlowGraphBitwiseAndBlock" /* FlowGraphBlockNames.BitwiseAnd */:
            return async () => (await __vitePreload(async () => { const {FlowGraphBitwiseAndBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphBitwiseAndBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphBitwiseAndBlock;
        case "FlowGraphBitwiseOrBlock" /* FlowGraphBlockNames.BitwiseOr */:
            return async () => (await __vitePreload(async () => { const {FlowGraphBitwiseOrBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphBitwiseOrBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphBitwiseOrBlock;
        case "FlowGraphBitwiseNotBlock" /* FlowGraphBlockNames.BitwiseNot */:
            return async () => (await __vitePreload(async () => { const {FlowGraphBitwiseNotBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphBitwiseNotBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphBitwiseNotBlock;
        case "FlowGraphBitwiseXorBlock" /* FlowGraphBlockNames.BitwiseXor */:
            return async () => (await __vitePreload(async () => { const {FlowGraphBitwiseXorBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphBitwiseXorBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphBitwiseXorBlock;
        case "FlowGraphBitwiseLeftShiftBlock" /* FlowGraphBlockNames.BitwiseLeftShift */:
            return async () => (await __vitePreload(async () => { const {FlowGraphBitwiseLeftShiftBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphBitwiseLeftShiftBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphBitwiseLeftShiftBlock;
        case "FlowGraphBitwiseRightShiftBlock" /* FlowGraphBlockNames.BitwiseRightShift */:
            return async () => (await __vitePreload(async () => { const {FlowGraphBitwiseRightShiftBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphBitwiseRightShiftBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphBitwiseRightShiftBlock;
        case "FlowGraphLengthBlock" /* FlowGraphBlockNames.Length */:
            return async () => (await __vitePreload(async () => { const {FlowGraphLengthBlock} = await import('./flowGraphVectorMathBlocks-BMWlT9Au.js');return { FlowGraphLengthBlock }},true              ?__vite__mapDeps([20,1,2,3,4,17,18,19,7]):void 0,import.meta.url)).FlowGraphLengthBlock;
        case "FlowGraphNormalizeBlock" /* FlowGraphBlockNames.Normalize */:
            return async () => (await __vitePreload(async () => { const {FlowGraphNormalizeBlock} = await import('./flowGraphVectorMathBlocks-BMWlT9Au.js');return { FlowGraphNormalizeBlock }},true              ?__vite__mapDeps([20,1,2,3,4,17,18,19,7]):void 0,import.meta.url)).FlowGraphNormalizeBlock;
        case "FlowGraphDotBlock" /* FlowGraphBlockNames.Dot */:
            return async () => (await __vitePreload(async () => { const {FlowGraphDotBlock} = await import('./flowGraphVectorMathBlocks-BMWlT9Au.js');return { FlowGraphDotBlock }},true              ?__vite__mapDeps([20,1,2,3,4,17,18,19,7]):void 0,import.meta.url)).FlowGraphDotBlock;
        case "FlowGraphCrossBlock" /* FlowGraphBlockNames.Cross */:
            return async () => (await __vitePreload(async () => { const {FlowGraphCrossBlock} = await import('./flowGraphVectorMathBlocks-BMWlT9Au.js');return { FlowGraphCrossBlock }},true              ?__vite__mapDeps([20,1,2,3,4,17,18,19,7]):void 0,import.meta.url)).FlowGraphCrossBlock;
        case "FlowGraphRotate2DBlock" /* FlowGraphBlockNames.Rotate2D */:
            return async () => (await __vitePreload(async () => { const {FlowGraphRotate2DBlock} = await import('./flowGraphVectorMathBlocks-BMWlT9Au.js');return { FlowGraphRotate2DBlock }},true              ?__vite__mapDeps([20,1,2,3,4,17,18,19,7]):void 0,import.meta.url)).FlowGraphRotate2DBlock;
        case "FlowGraphRotate3DBlock" /* FlowGraphBlockNames.Rotate3D */:
            return async () => (await __vitePreload(async () => { const {FlowGraphRotate3DBlock} = await import('./flowGraphVectorMathBlocks-BMWlT9Au.js');return { FlowGraphRotate3DBlock }},true              ?__vite__mapDeps([20,1,2,3,4,17,18,19,7]):void 0,import.meta.url)).FlowGraphRotate3DBlock;
        case "FlowGraphTransposeBlock" /* FlowGraphBlockNames.Transpose */:
            return async () => (await __vitePreload(async () => { const {FlowGraphTransposeBlock} = await import('./flowGraphMatrixMathBlocks-CVA57ClY.js');return { FlowGraphTransposeBlock }},true              ?__vite__mapDeps([21,1,2,3,4,19,18,17,7]):void 0,import.meta.url)).FlowGraphTransposeBlock;
        case "FlowGraphDeterminantBlock" /* FlowGraphBlockNames.Determinant */:
            return async () => (await __vitePreload(async () => { const {FlowGraphDeterminantBlock} = await import('./flowGraphMatrixMathBlocks-CVA57ClY.js');return { FlowGraphDeterminantBlock }},true              ?__vite__mapDeps([21,1,2,3,4,19,18,17,7]):void 0,import.meta.url)).FlowGraphDeterminantBlock;
        case "FlowGraphInvertMatrixBlock" /* FlowGraphBlockNames.InvertMatrix */:
            return async () => (await __vitePreload(async () => { const {FlowGraphInvertMatrixBlock} = await import('./flowGraphMatrixMathBlocks-CVA57ClY.js');return { FlowGraphInvertMatrixBlock }},true              ?__vite__mapDeps([21,1,2,3,4,19,18,17,7]):void 0,import.meta.url)).FlowGraphInvertMatrixBlock;
        case "FlowGraphMatrixMultiplicationBlock" /* FlowGraphBlockNames.MatrixMultiplication */:
            return async () => (await __vitePreload(async () => { const {FlowGraphMatrixMultiplicationBlock} = await import('./flowGraphMatrixMathBlocks-CVA57ClY.js');return { FlowGraphMatrixMultiplicationBlock }},true              ?__vite__mapDeps([21,1,2,3,4,19,18,17,7]):void 0,import.meta.url)).FlowGraphMatrixMultiplicationBlock;
        case "FlowGraphBranchBlock" /* FlowGraphBlockNames.Branch */:
            return async () => (await __vitePreload(async () => { const {FlowGraphBranchBlock} = await import('./flowGraphBranchBlock-DiLZOxxs.js');return { FlowGraphBranchBlock }},true              ?__vite__mapDeps([22,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphBranchBlock;
        case "FlowGraphSetDelayBlock" /* FlowGraphBlockNames.SetDelay */:
            return async () => (await __vitePreload(async () => { const {FlowGraphSetDelayBlock} = await import('./flowGraphSetDelayBlock-BH-Ttv2A.js');return { FlowGraphSetDelayBlock }},true              ?__vite__mapDeps([23,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphSetDelayBlock;
        case "FlowGraphCancelDelayBlock" /* FlowGraphBlockNames.CancelDelay */:
            return async () => (await __vitePreload(async () => { const {FlowGraphCancelDelayBlock} = await import('./flowGraphCancelDelayBlock-DdnCJIHp.js');return { FlowGraphCancelDelayBlock }},true              ?__vite__mapDeps([24,2,3,4,1,7]):void 0,import.meta.url)).FlowGraphCancelDelayBlock;
        case "FlowGraphCallCounterBlock" /* FlowGraphBlockNames.CallCounter */:
            return async () => (await __vitePreload(async () => { const {FlowGraphCallCounterBlock} = await import('./flowGraphCounterBlock-CHfEWEKo.js');return { FlowGraphCallCounterBlock }},true              ?__vite__mapDeps([25,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphCallCounterBlock;
        case "FlowGraphDebounceBlock" /* FlowGraphBlockNames.Debounce */:
            return async () => (await __vitePreload(async () => { const {FlowGraphDebounceBlock} = await import('./flowGraphDebounceBlock-B7BN833I.js');return { FlowGraphDebounceBlock }},true              ?__vite__mapDeps([26,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphDebounceBlock;
        case "FlowGraphThrottleBlock" /* FlowGraphBlockNames.Throttle */:
            return async () => (await __vitePreload(async () => { const {FlowGraphThrottleBlock} = await import('./flowGraphThrottleBlock-xEOLL07m.js');return { FlowGraphThrottleBlock }},true              ?__vite__mapDeps([27,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphThrottleBlock;
        case "FlowGraphDoNBlock" /* FlowGraphBlockNames.DoN */:
            return async () => (await __vitePreload(async () => { const {FlowGraphDoNBlock} = await import('./flowGraphDoNBlock-BsIGVnlM.js');return { FlowGraphDoNBlock }},true              ?__vite__mapDeps([28,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphDoNBlock;
        case "FlowGraphFlipFlopBlock" /* FlowGraphBlockNames.FlipFlop */:
            return async () => (await __vitePreload(async () => { const {FlowGraphFlipFlopBlock} = await import('./flowGraphFlipFlopBlock-CvA24zJ2.js');return { FlowGraphFlipFlopBlock }},true              ?__vite__mapDeps([29,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphFlipFlopBlock;
        case "FlowGraphForLoopBlock" /* FlowGraphBlockNames.ForLoop */:
            return async () => (await __vitePreload(async () => { const {FlowGraphForLoopBlock} = await import('./flowGraphForLoopBlock-objpPanS.js');return { FlowGraphForLoopBlock }},true              ?__vite__mapDeps([30,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphForLoopBlock;
        case "FlowGraphMultiGateBlock" /* FlowGraphBlockNames.MultiGate */:
            return async () => (await __vitePreload(async () => { const {FlowGraphMultiGateBlock} = await import('./flowGraphMultiGateBlock-DiPiWOHH.js');return { FlowGraphMultiGateBlock }},true              ?__vite__mapDeps([31,2,3,4,1,7]):void 0,import.meta.url)).FlowGraphMultiGateBlock;
        case "FlowGraphSequenceBlock" /* FlowGraphBlockNames.Sequence */:
            return async () => (await __vitePreload(async () => { const {FlowGraphSequenceBlock} = await import('./flowGraphSequenceBlock-B0Rrk3RL.js');return { FlowGraphSequenceBlock }},true              ?__vite__mapDeps([32,2,3,4,1,7]):void 0,import.meta.url)).FlowGraphSequenceBlock;
        case "FlowGraphSwitchBlock" /* FlowGraphBlockNames.Switch */:
            return async () => (await __vitePreload(async () => { const {FlowGraphSwitchBlock} = await import('./flowGraphSwitchBlock-twUIuJlv.js');return { FlowGraphSwitchBlock }},true              ?__vite__mapDeps([33,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphSwitchBlock;
        case "FlowGraphWaitAllBlock" /* FlowGraphBlockNames.WaitAll */:
            return async () => (await __vitePreload(async () => { const {FlowGraphWaitAllBlock} = await import('./flowGraphWaitAllBlock-m7YQzoZP.js');return { FlowGraphWaitAllBlock }},true              ?__vite__mapDeps([34,2,3,4,1,7]):void 0,import.meta.url)).FlowGraphWaitAllBlock;
        case "FlowGraphWhileLoopBlock" /* FlowGraphBlockNames.WhileLoop */:
            return async () => (await __vitePreload(async () => { const {FlowGraphWhileLoopBlock} = await import('./flowGraphWhileLoopBlock-DzQfdud0.js');return { FlowGraphWhileLoopBlock }},true              ?__vite__mapDeps([35,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphWhileLoopBlock;
        case "FlowGraphConsoleLogBlock" /* FlowGraphBlockNames.ConsoleLog */:
            return async () => (await __vitePreload(async () => { const {FlowGraphConsoleLogBlock} = await import('./flowGraphConsoleLogBlock-C6T6-DAQ.js');return { FlowGraphConsoleLogBlock }},true              ?__vite__mapDeps([36,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphConsoleLogBlock;
        case "FlowGraphConditionalBlock" /* FlowGraphBlockNames.Conditional */:
            return async () => (await __vitePreload(async () => { const {FlowGraphConditionalDataBlock} = await import('./flowGraphConditionalDataBlock-C-UNv5MV.js');return { FlowGraphConditionalDataBlock }},true              ?__vite__mapDeps([37,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphConditionalDataBlock;
        case "FlowGraphConstantBlock" /* FlowGraphBlockNames.Constant */:
            return async () => (await __vitePreload(async () => { const {FlowGraphConstantBlock} = await import('./flowGraphConstantBlock-Cd6_VUbG.js');return { FlowGraphConstantBlock }},true              ?__vite__mapDeps([38,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphConstantBlock;
        case "FlowGraphTransformCoordinatesSystemBlock" /* FlowGraphBlockNames.TransformCoordinatesSystem */:
            return async () => (await __vitePreload(async () => { const {FlowGraphTransformCoordinatesSystemBlock} = await import('./flowGraphTransformCoordinatesSystemBlock-Ci41pAjS.js');return { FlowGraphTransformCoordinatesSystemBlock }},true              ?__vite__mapDeps([39,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphTransformCoordinatesSystemBlock;
        case "FlowGraphGetAssetBlock" /* FlowGraphBlockNames.GetAsset */:
            return async () => (await __vitePreload(async () => { const {FlowGraphGetAssetBlock} = await import('./flowGraphGetAssetBlock-eRHQvRSt.js');return { FlowGraphGetAssetBlock }},true              ?__vite__mapDeps([40,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphGetAssetBlock;
        case "FlowGraphGetPropertyBlock" /* FlowGraphBlockNames.GetProperty */:
            return async () => (await __vitePreload(async () => { const {FlowGraphGetPropertyBlock} = await import('./flowGraphGetPropertyBlock-B2jLqZtA.js');return { FlowGraphGetPropertyBlock }},true              ?__vite__mapDeps([41,1,2,3,4,18,7]):void 0,import.meta.url)).FlowGraphGetPropertyBlock;
        case "FlowGraphSetPropertyBlock" /* FlowGraphBlockNames.SetProperty */:
            return async () => (await __vitePreload(async () => { const {FlowGraphSetPropertyBlock} = await import('./flowGraphSetPropertyBlock-C412YpUC.js');return { FlowGraphSetPropertyBlock }},true              ?__vite__mapDeps([42,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphSetPropertyBlock;
        case "FlowGraphGetVariableBlock" /* FlowGraphBlockNames.GetVariable */:
            return async () => (await __vitePreload(async () => { const {FlowGraphGetVariableBlock} = await import('./flowGraphGetVariableBlock-D91Cgc9v.js');return { FlowGraphGetVariableBlock }},true              ?__vite__mapDeps([43,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphGetVariableBlock;
        case "FlowGraphSetVariableBlock" /* FlowGraphBlockNames.SetVariable */:
            return async () => (await __vitePreload(async () => { const {FlowGraphSetVariableBlock} = await import('./flowGraphSetVariableBlock-Bp8960gb.js');return { FlowGraphSetVariableBlock }},true              ?__vite__mapDeps([44,2,3,4,1,7]):void 0,import.meta.url)).FlowGraphSetVariableBlock;
        case "FlowGraphJsonPointerParserBlock" /* FlowGraphBlockNames.JsonPointerParser */:
            return async () => (await __vitePreload(async () => { const {FlowGraphJsonPointerParserBlock} = await import('./flowGraphJsonPointerParserBlock-BkiHU-aD.js');return { FlowGraphJsonPointerParserBlock }},true              ?__vite__mapDeps([45,1,2,3,4,18,7]):void 0,import.meta.url)).FlowGraphJsonPointerParserBlock;
        case "FlowGraphLeadingZerosBlock" /* FlowGraphBlockNames.LeadingZeros */:
            return async () => (await __vitePreload(async () => { const {FlowGraphLeadingZerosBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphLeadingZerosBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphLeadingZerosBlock;
        case "FlowGraphTrailingZerosBlock" /* FlowGraphBlockNames.TrailingZeros */:
            return async () => (await __vitePreload(async () => { const {FlowGraphTrailingZerosBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphTrailingZerosBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphTrailingZerosBlock;
        case "FlowGraphOneBitsCounterBlock" /* FlowGraphBlockNames.OneBitsCounter */:
            return async () => (await __vitePreload(async () => { const {FlowGraphOneBitsCounterBlock} = await import('./flowGraphMathBlocks-CBHd7jx3.js');return { FlowGraphOneBitsCounterBlock }},true              ?__vite__mapDeps([16,2,3,4,1,17,18,19,7]):void 0,import.meta.url)).FlowGraphOneBitsCounterBlock;
        case "FlowGraphCombineVector2Block" /* FlowGraphBlockNames.CombineVector2 */:
            return async () => (await __vitePreload(async () => { const {FlowGraphCombineVector2Block} = await import('./flowGraphMathCombineExtractBlocks-qEm2nMP2.js');return { FlowGraphCombineVector2Block }},true              ?__vite__mapDeps([46,18,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphCombineVector2Block;
        case "FlowGraphCombineVector3Block" /* FlowGraphBlockNames.CombineVector3 */:
            return async () => (await __vitePreload(async () => { const {FlowGraphCombineVector3Block} = await import('./flowGraphMathCombineExtractBlocks-qEm2nMP2.js');return { FlowGraphCombineVector3Block }},true              ?__vite__mapDeps([46,18,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphCombineVector3Block;
        case "FlowGraphCombineVector4Block" /* FlowGraphBlockNames.CombineVector4 */:
            return async () => (await __vitePreload(async () => { const {FlowGraphCombineVector4Block} = await import('./flowGraphMathCombineExtractBlocks-qEm2nMP2.js');return { FlowGraphCombineVector4Block }},true              ?__vite__mapDeps([46,18,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphCombineVector4Block;
        case "FlowGraphCombineMatrixBlock" /* FlowGraphBlockNames.CombineMatrix */:
            return async () => (await __vitePreload(async () => { const {FlowGraphCombineMatrixBlock} = await import('./flowGraphMathCombineExtractBlocks-qEm2nMP2.js');return { FlowGraphCombineMatrixBlock }},true              ?__vite__mapDeps([46,18,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphCombineMatrixBlock;
        case "FlowGraphExtractVector2Block" /* FlowGraphBlockNames.ExtractVector2 */:
            return async () => (await __vitePreload(async () => { const {FlowGraphExtractVector2Block} = await import('./flowGraphMathCombineExtractBlocks-qEm2nMP2.js');return { FlowGraphExtractVector2Block }},true              ?__vite__mapDeps([46,18,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphExtractVector2Block;
        case "FlowGraphExtractVector3Block" /* FlowGraphBlockNames.ExtractVector3 */:
            return async () => (await __vitePreload(async () => { const {FlowGraphExtractVector3Block} = await import('./flowGraphMathCombineExtractBlocks-qEm2nMP2.js');return { FlowGraphExtractVector3Block }},true              ?__vite__mapDeps([46,18,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphExtractVector3Block;
        case "FlowGraphExtractVector4Block" /* FlowGraphBlockNames.ExtractVector4 */:
            return async () => (await __vitePreload(async () => { const {FlowGraphExtractVector4Block} = await import('./flowGraphMathCombineExtractBlocks-qEm2nMP2.js');return { FlowGraphExtractVector4Block }},true              ?__vite__mapDeps([46,18,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphExtractVector4Block;
        case "FlowGraphExtractMatrixBlock" /* FlowGraphBlockNames.ExtractMatrix */:
            return async () => (await __vitePreload(async () => { const {FlowGraphExtractMatrixBlock} = await import('./flowGraphMathCombineExtractBlocks-qEm2nMP2.js');return { FlowGraphExtractMatrixBlock }},true              ?__vite__mapDeps([46,18,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphExtractMatrixBlock;
        case "FlowGraphTransformVectorBlock" /* FlowGraphBlockNames.TransformVector */:
            return async () => (await __vitePreload(async () => { const {FlowGraphTransformBlock} = await import('./flowGraphVectorMathBlocks-BMWlT9Au.js');return { FlowGraphTransformBlock }},true              ?__vite__mapDeps([20,1,2,3,4,17,18,19,7]):void 0,import.meta.url)).FlowGraphTransformBlock;
        case "FlowGraphTransformCoordinatesBlock" /* FlowGraphBlockNames.TransformCoordinates */:
            return async () => (await __vitePreload(async () => { const {FlowGraphTransformCoordinatesBlock} = await import('./flowGraphVectorMathBlocks-BMWlT9Au.js');return { FlowGraphTransformCoordinatesBlock }},true              ?__vite__mapDeps([20,1,2,3,4,17,18,19,7]):void 0,import.meta.url)).FlowGraphTransformCoordinatesBlock;
        case "FlowGraphConjugateBlock" /* FlowGraphBlockNames.Conjugate */:
            return async () => (await __vitePreload(async () => { const {FlowGraphConjugateBlock} = await import('./flowGraphVectorMathBlocks-BMWlT9Au.js');return { FlowGraphConjugateBlock }},true              ?__vite__mapDeps([20,1,2,3,4,17,18,19,7]):void 0,import.meta.url)).FlowGraphConjugateBlock;
        case "FlowGraphAngleBetweenBlock" /* FlowGraphBlockNames.AngleBetween */:
            return async () => (await __vitePreload(async () => { const {FlowGraphAngleBetweenBlock} = await import('./flowGraphVectorMathBlocks-BMWlT9Au.js');return { FlowGraphAngleBetweenBlock }},true              ?__vite__mapDeps([20,1,2,3,4,17,18,19,7]):void 0,import.meta.url)).FlowGraphAngleBetweenBlock;
        case "FlowGraphQuaternionFromAxisAngleBlock" /* FlowGraphBlockNames.QuaternionFromAxisAngle */:
            return async () => (await __vitePreload(async () => { const {FlowGraphQuaternionFromAxisAngleBlock} = await import('./flowGraphVectorMathBlocks-BMWlT9Au.js');return { FlowGraphQuaternionFromAxisAngleBlock }},true              ?__vite__mapDeps([20,1,2,3,4,17,18,19,7]):void 0,import.meta.url)).FlowGraphQuaternionFromAxisAngleBlock;
        case "FlowGraphAxisAngleFromQuaternionBlock" /* FlowGraphBlockNames.AxisAngleFromQuaternion */:
            return async () => (await __vitePreload(async () => { const {FlowGraphAxisAngleFromQuaternionBlock} = await import('./flowGraphVectorMathBlocks-BMWlT9Au.js');return { FlowGraphAxisAngleFromQuaternionBlock }},true              ?__vite__mapDeps([20,1,2,3,4,17,18,19,7]):void 0,import.meta.url)).FlowGraphAxisAngleFromQuaternionBlock;
        case "FlowGraphQuaternionFromDirectionsBlock" /* FlowGraphBlockNames.QuaternionFromDirections */:
            return async () => (await __vitePreload(async () => { const {FlowGraphQuaternionFromDirectionsBlock} = await import('./flowGraphVectorMathBlocks-BMWlT9Au.js');return { FlowGraphQuaternionFromDirectionsBlock }},true              ?__vite__mapDeps([20,1,2,3,4,17,18,19,7]):void 0,import.meta.url)).FlowGraphQuaternionFromDirectionsBlock;
        case "FlowGraphMatrixDecompose" /* FlowGraphBlockNames.MatrixDecompose */:
            return async () => (await __vitePreload(async () => { const {FlowGraphMatrixDecomposeBlock} = await import('./flowGraphMatrixMathBlocks-CVA57ClY.js');return { FlowGraphMatrixDecomposeBlock }},true              ?__vite__mapDeps([21,1,2,3,4,19,18,17,7]):void 0,import.meta.url)).FlowGraphMatrixDecomposeBlock;
        case "FlowGraphMatrixCompose" /* FlowGraphBlockNames.MatrixCompose */:
            return async () => (await __vitePreload(async () => { const {FlowGraphMatrixComposeBlock} = await import('./flowGraphMatrixMathBlocks-CVA57ClY.js');return { FlowGraphMatrixComposeBlock }},true              ?__vite__mapDeps([21,1,2,3,4,19,18,17,7]):void 0,import.meta.url)).FlowGraphMatrixComposeBlock;
        case "FlowGraphBooleanToFloat" /* FlowGraphBlockNames.BooleanToFloat */:
            return async () => (await __vitePreload(async () => { const {FlowGraphBooleanToFloat} = await import('./flowGraphTypeToTypeBlocks-BCIt2CE0.js');return { FlowGraphBooleanToFloat }},true              ?__vite__mapDeps([47,19,18,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphBooleanToFloat;
        case "FlowGraphBooleanToInt" /* FlowGraphBlockNames.BooleanToInt */:
            return async () => (await __vitePreload(async () => { const {FlowGraphBooleanToInt} = await import('./flowGraphTypeToTypeBlocks-BCIt2CE0.js');return { FlowGraphBooleanToInt }},true              ?__vite__mapDeps([47,19,18,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphBooleanToInt;
        case "FlowGraphFloatToBoolean" /* FlowGraphBlockNames.FloatToBoolean */:
            return async () => (await __vitePreload(async () => { const {FlowGraphFloatToBoolean} = await import('./flowGraphTypeToTypeBlocks-BCIt2CE0.js');return { FlowGraphFloatToBoolean }},true              ?__vite__mapDeps([47,19,18,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphFloatToBoolean;
        case "FlowGraphIntToBoolean" /* FlowGraphBlockNames.IntToBoolean */:
            return async () => (await __vitePreload(async () => { const {FlowGraphIntToBoolean} = await import('./flowGraphTypeToTypeBlocks-BCIt2CE0.js');return { FlowGraphIntToBoolean }},true              ?__vite__mapDeps([47,19,18,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphIntToBoolean;
        case "FlowGraphIntToFloat" /* FlowGraphBlockNames.IntToFloat */:
            return async () => (await __vitePreload(async () => { const {FlowGraphIntToFloat} = await import('./flowGraphTypeToTypeBlocks-BCIt2CE0.js');return { FlowGraphIntToFloat }},true              ?__vite__mapDeps([47,19,18,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphIntToFloat;
        case "FlowGraphFloatToInt" /* FlowGraphBlockNames.FloatToInt */:
            return async () => (await __vitePreload(async () => { const {FlowGraphFloatToInt} = await import('./flowGraphTypeToTypeBlocks-BCIt2CE0.js');return { FlowGraphFloatToInt }},true              ?__vite__mapDeps([47,19,18,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphFloatToInt;
        case "FlowGraphEasingBlock" /* FlowGraphBlockNames.Easing */:
            return async () => (await __vitePreload(async () => { const {FlowGraphEasingBlock} = await import('./flowGraphEasingBlock-DPpBHrfT.js');return { FlowGraphEasingBlock }},true              ?__vite__mapDeps([48,2,3,4,1,7]):void 0,import.meta.url)).FlowGraphEasingBlock;
        case "FlowGraphBezierCurveEasing" /* FlowGraphBlockNames.BezierCurveEasing */:
            return async () => (await __vitePreload(async () => { const {FlowGraphBezierCurveEasingBlock} = await import('./flowGraphBezierCurveEasingBlock-Q5r3yjWs.js');return { FlowGraphBezierCurveEasingBlock }},true              ?__vite__mapDeps([49,2,3,4,1,7]):void 0,import.meta.url)).FlowGraphBezierCurveEasingBlock;
        case "FlowGraphPointerOverEventBlock" /* FlowGraphBlockNames.PointerOverEvent */:
            return async () => (await __vitePreload(async () => { const {FlowGraphPointerOverEventBlock} = await import('./flowGraphPointerOverEventBlock-DMmP4NMG.js');return { FlowGraphPointerOverEventBlock }},true              ?__vite__mapDeps([50,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphPointerOverEventBlock;
        case "FlowGraphPointerOutEventBlock" /* FlowGraphBlockNames.PointerOutEvent */:
            return async () => (await __vitePreload(async () => { const {FlowGraphPointerOutEventBlock} = await import('./flowGraphPointerOutEventBlock-BTSYlAKI.js');return { FlowGraphPointerOutEventBlock }},true              ?__vite__mapDeps([51,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphPointerOutEventBlock;
        case "FlowGraphContextBlock" /* FlowGraphBlockNames.Context */:
            return async () => (await __vitePreload(async () => { const {FlowGraphContextBlock} = await import('./flowGraphContextBlock-D4tc8TQP.js');return { FlowGraphContextBlock }},true              ?__vite__mapDeps([52,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphContextBlock;
        case "FlowGraphArrayIndexBlock" /* FlowGraphBlockNames.ArrayIndex */:
            return async () => (await __vitePreload(async () => { const {FlowGraphArrayIndexBlock} = await import('./flowGraphArrayIndexBlock-NJcLU5dO.js');return { FlowGraphArrayIndexBlock }},true              ?__vite__mapDeps([53,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphArrayIndexBlock;
        case "FlowGraphCodeExecutionBlock" /* FlowGraphBlockNames.CodeExecution */:
            return async () => (await __vitePreload(async () => { const {FlowGraphCodeExecutionBlock} = await import('./flowGraphCodeExecutionBlock-D2nzjSIh.js');return { FlowGraphCodeExecutionBlock }},true              ?__vite__mapDeps([54,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphCodeExecutionBlock;
        case "FlowGraphIndexOfBlock" /* FlowGraphBlockNames.IndexOf */:
            return async () => (await __vitePreload(async () => { const {FlowGraphIndexOfBlock} = await import('./flowGraphIndexOfBlock-EHh5o3oq.js');return { FlowGraphIndexOfBlock }},true              ?__vite__mapDeps([55,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphIndexOfBlock;
        case "FlowGraphFunctionReference" /* FlowGraphBlockNames.FunctionReference */:
            return async () => (await __vitePreload(async () => { const {FlowGraphFunctionReferenceBlock} = await import('./flowGraphFunctionReferenceBlock-ByGpsFPE.js');return { FlowGraphFunctionReferenceBlock }},true              ?__vite__mapDeps([56,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphFunctionReferenceBlock;
        case "FlowGraphDataSwitchBlock" /* FlowGraphBlockNames.DataSwitch */:
            return async () => (await __vitePreload(async () => { const {FlowGraphDataSwitchBlock} = await import('./flowGraphDataSwitchBlock-DSObniwd.js');return { FlowGraphDataSwitchBlock }},true              ?__vite__mapDeps([57,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphDataSwitchBlock;
        default:
            // check if the block is a custom block
            if (CustomBlocks[blockName]) {
                return CustomBlocks[blockName];
            }
            throw new Error(`Unknown block name ${blockName}`);
    }
}

/**
 * An execution block that has an out signal. This signal is triggered when the synchronous execution of this block is done.
 * Most execution blocks will inherit from this, except for the ones that have multiple signals to be triggered.
 * (such as if blocks)
 */
class FlowGraphExecutionBlockWithOutSignal extends FlowGraphExecutionBlock {
    constructor(config) {
        super(config);
        this.out = this._registerSignalOutput("out");
    }
}

/**
 * An async execution block can start tasks that will be executed asynchronously.
 * It should also be responsible for clearing it in _cancelPendingTasks.
 */
class FlowGraphAsyncExecutionBlock extends FlowGraphExecutionBlockWithOutSignal {
    constructor(config, events) {
        super(config);
        this._eventsSignalOutputs = {};
        this.done = this._registerSignalOutput("done");
        if (events) {
            for (const eventName of events) {
                this._eventsSignalOutputs[eventName] = this._registerSignalOutput(eventName + "Event");
            }
        }
    }
    /**
     * @internal
     * This function can be overridden to execute any
     * logic that should be executed on every frame
     * while the async task is pending.
     * @param context the context in which it is running
     */
    _executeOnTick(_context) { }
    /**
     * @internal
     * @param context
     */
    _startPendingTasks(context) {
        if (context._getExecutionVariable(this, "_initialized", false)) {
            this._cancelPendingTasks(context);
            this._resetAfterCanceled(context);
        }
        this._preparePendingTasks(context);
        context._addPendingBlock(this);
        this.out._activateSignal(context);
        context._setExecutionVariable(this, "_initialized", true);
    }
    _resetAfterCanceled(context) {
        context._deleteExecutionVariable(this, "_initialized");
        context._removePendingBlock(this);
    }
}

/**
 * A type of block that listens to an event observable and activates
 * its output signal when the event is triggered.
 */
class FlowGraphEventBlock extends FlowGraphAsyncExecutionBlock {
    constructor() {
        super(...arguments);
        /**
         * the priority of initialization of this block.
         * For example, scene start should have a negative priority because it should be initialized last.
         */
        this.initPriority = 0;
        /**
         * The type of the event
         */
        this.type = "NoTrigger" /* FlowGraphEventType.NoTrigger */;
    }
    /**
     * @internal
     */
    _execute(context) {
        context._notifyExecuteNode(this);
        this.done._activateSignal(context);
    }
}

/**
 * Given a list of blocks, find an output data connection that has a specific unique id
 * @param blocks a list of flow graph blocks
 * @param uniqueId the unique id of a connection
 * @returns the connection that has this unique id. throws an error if none was found
 */
function GetDataOutConnectionByUniqueId(blocks, uniqueId) {
    for (const block of blocks) {
        for (const dataOut of block.dataOutputs) {
            if (dataOut.uniqueId === uniqueId) {
                return dataOut;
            }
        }
    }
    throw new Error("Could not find data out connection with unique id " + uniqueId);
}
/**
 * Given a list of blocks, find an input signal connection that has a specific unique id
 * @param blocks a list of flow graph blocks
 * @param uniqueId the unique id of a connection
 * @returns the connection that has this unique id. throws an error if none was found
 */
function GetSignalInConnectionByUniqueId(blocks, uniqueId) {
    for (const block of blocks) {
        if (block instanceof FlowGraphExecutionBlock) {
            for (const signalIn of block.signalInputs) {
                if (signalIn.uniqueId === uniqueId) {
                    return signalIn;
                }
            }
        }
    }
    throw new Error("Could not find signal in connection with unique id " + uniqueId);
}
/**
 * Parses a graph from a given serialization object
 * @param serializationObject the object where the values are written
 * @param options options for parsing the graph
 * @returns the parsed graph
 */
async function ParseFlowGraphAsync(serializationObject, options) {
    // get all classes types needed for the blocks using the block factory
    const resolvedClasses = await Promise.all(serializationObject.allBlocks.map(async (serializedBlock) => {
        const classFactory = blockFactory(serializedBlock.className);
        return await classFactory();
    }));
    // async will be used when we start using the block async factory
    return ParseFlowGraph(serializationObject, options, resolvedClasses);
}
/**
 * Parses a graph from a given serialization object
 * @param serializationObject the object where the values are written
 * @param options options for parsing the graph
 * @param resolvedClasses the resolved classes for the blocks
 * @returns the parsed graph
 */
function ParseFlowGraph(serializationObject, options, resolvedClasses) {
    const graph = options.coordinator.createGraph();
    const blocks = [];
    const valueParseFunction = options.valueParseFunction ?? defaultValueParseFunction;
    // Parse all blocks
    // for (const serializedBlock of serializationObject.allBlocks) {
    for (let i = 0; i < serializationObject.allBlocks.length; i++) {
        const serializedBlock = serializationObject.allBlocks[i];
        const block = ParseFlowGraphBlockWithClassType(serializedBlock, { scene: options.coordinator.config.scene, pathConverter: options.pathConverter, assetsContainer: options.coordinator.config.scene, valueParseFunction }, resolvedClasses[i]);
        blocks.push(block);
        if (block instanceof FlowGraphEventBlock) {
            graph.addEventBlock(block);
        }
    }
    // After parsing all blocks, connect them
    for (const block of blocks) {
        for (const dataIn of block.dataInputs) {
            for (const serializedConnection of dataIn.connectedPointIds) {
                const connection = GetDataOutConnectionByUniqueId(blocks, serializedConnection);
                dataIn.connectTo(connection);
            }
        }
        if (block instanceof FlowGraphExecutionBlock) {
            for (const signalOut of block.signalOutputs) {
                for (const serializedConnection of signalOut.connectedPointIds) {
                    const connection = GetSignalInConnectionByUniqueId(blocks, serializedConnection);
                    signalOut.connectTo(connection);
                }
            }
        }
    }
    for (const serializedContext of serializationObject.executionContexts) {
        ParseFlowGraphContext(serializedContext, { graph, valueParseFunction }, serializationObject.rightHanded);
    }
    return graph;
}
/**
 * Parses a context
 * @param serializationObject the object containing the context serialization values
 * @param options the options for parsing the context
 * @param rightHanded whether the serialized data is right handed
 * @returns
 */
function ParseFlowGraphContext(serializationObject, options, rightHanded) {
    const result = options.graph.createContext();
    if (serializationObject.enableLogging) {
        result.enableLogging = true;
    }
    result.treatDataAsRightHanded = rightHanded || false;
    const valueParseFunction = options.valueParseFunction ?? defaultValueParseFunction;
    result.uniqueId = serializationObject.uniqueId;
    const scene = result.getScene();
    // check if assets context is available
    if (serializationObject._assetsContext) {
        const ac = serializationObject._assetsContext;
        const assetsContext = {
            meshes: ac.meshes?.map((m) => scene.getMeshById(m)),
            lights: ac.lights?.map((l) => scene.getLightByName(l)),
            cameras: ac.cameras?.map((c) => scene.getCameraByName(c)),
            materials: ac.materials?.map((m) => scene.getMaterialById(m)),
            textures: ac.textures?.map((t) => scene.getTextureByName(t)),
            animations: ac.animations?.map((a) => scene.animations.find((anim) => anim.name === a)),
            skeletons: ac.skeletons?.map((s) => scene.getSkeletonByName(s)),
            particleSystems: ac.particleSystems?.map((ps) => scene.getParticleSystemById(ps)),
            animationGroups: ac.animationGroups?.map((ag) => scene.getAnimationGroupByName(ag)),
            transformNodes: ac.transformNodes?.map((tn) => scene.getTransformNodeById(tn)),
            rootNodes: [],
            multiMaterials: [],
            morphTargetManagers: [],
            geometries: [],
            actionManagers: [],
            environmentTexture: null,
            postProcesses: [],
            sounds: null,
            effectLayers: [],
            layers: [],
            reflectionProbes: [],
            lensFlareSystems: [],
            proceduralTextures: [],
            getNodes: function () {
                throw new Error("Function not implemented.");
            },
        };
        result.assetsContext = assetsContext;
    }
    for (const key in serializationObject._userVariables) {
        const value = valueParseFunction(key, serializationObject._userVariables, result.assetsContext, scene);
        result.userVariables[key] = value;
    }
    for (const key in serializationObject._connectionValues) {
        const value = valueParseFunction(key, serializationObject._connectionValues, result.assetsContext, scene);
        result._setConnectionValueByKey(key, value);
    }
    return result;
}
/**
 * Parses a block from a serialization object
 * @param serializationObject the object to parse from
 * @param parseOptions options for parsing the block
 * @param classType the class type of the block. This is used when the class is not loaded asynchronously
 * @returns the parsed block
 */
function ParseFlowGraphBlockWithClassType(serializationObject, parseOptions, classType) {
    const parsedConfig = {};
    const valueParseFunction = parseOptions.valueParseFunction ?? defaultValueParseFunction;
    if (serializationObject.config) {
        for (const key in serializationObject.config) {
            parsedConfig[key] = valueParseFunction(key, serializationObject.config, parseOptions.assetsContainer || parseOptions.scene, parseOptions.scene);
        }
    }
    if (needsPathConverter(serializationObject.className)) {
        if (!parseOptions.pathConverter) {
            throw new Error("Path converter is required for this block");
        }
        parsedConfig.pathConverter = parseOptions.pathConverter;
    }
    const obj = new classType(parsedConfig);
    obj.uniqueId = serializationObject.uniqueId;
    for (let i = 0; i < serializationObject.dataInputs.length; i++) {
        const dataInput = obj.getDataInput(serializationObject.dataInputs[i].name);
        if (dataInput) {
            dataInput.deserialize(serializationObject.dataInputs[i]);
        }
        else {
            throw new Error("Could not find data input with name " + serializationObject.dataInputs[i].name + " in block " + serializationObject.className);
        }
    }
    for (let i = 0; i < serializationObject.dataOutputs.length; i++) {
        const dataOutput = obj.getDataOutput(serializationObject.dataOutputs[i].name);
        if (dataOutput) {
            dataOutput.deserialize(serializationObject.dataOutputs[i]);
        }
        else {
            throw new Error("Could not find data output with name " + serializationObject.dataOutputs[i].name + " in block " + serializationObject.className);
        }
    }
    obj.metadata = serializationObject.metadata;
    obj.deserialize && obj.deserialize(serializationObject);
    return obj;
}

// eslint-disable-next-line @typescript-eslint/naming-convention
const gltfTypeToBabylonType = {
    float: { length: 1, flowGraphType: "number" /* FlowGraphTypes.Number */, elementType: "number" },
    bool: { length: 1, flowGraphType: "boolean" /* FlowGraphTypes.Boolean */, elementType: "boolean" },
    float2: { length: 2, flowGraphType: "Vector2" /* FlowGraphTypes.Vector2 */, elementType: "number" },
    float3: { length: 3, flowGraphType: "Vector3" /* FlowGraphTypes.Vector3 */, elementType: "number" },
    float4: { length: 4, flowGraphType: "Vector4" /* FlowGraphTypes.Vector4 */, elementType: "number" },
    float4x4: { length: 16, flowGraphType: "Matrix" /* FlowGraphTypes.Matrix */, elementType: "number" },
    float2x2: { length: 4, flowGraphType: "Matrix2D" /* FlowGraphTypes.Matrix2D */, elementType: "number" },
    float3x3: { length: 9, flowGraphType: "Matrix3D" /* FlowGraphTypes.Matrix3D */, elementType: "number" },
    int: { length: 1, flowGraphType: "FlowGraphInteger" /* FlowGraphTypes.Integer */, elementType: "number" },
};
class InteractivityGraphToFlowGraphParser {
    constructor(_interactivityGraph, _gltf, _animationTargetFps = 60) {
        this._interactivityGraph = _interactivityGraph;
        this._gltf = _gltf;
        this._animationTargetFps = _animationTargetFps;
        /**
         * Note - the graph should be rejected if the same type is defined twice.
         * We currently don't validate that.
         */
        this._types = [];
        this._mappings = [];
        this._staticVariables = [];
        this._events = [];
        this._internalEventsCounter = 0;
        this._nodes = [];
        // start with types
        this._parseTypes();
        // continue with declarations
        this._parseDeclarations();
        this._parseVariables();
        this._parseEvents();
        this._parseNodes();
    }
    get arrays() {
        return {
            types: this._types,
            mappings: this._mappings,
            staticVariables: this._staticVariables,
            events: this._events,
            nodes: this._nodes,
        };
    }
    _parseTypes() {
        if (!this._interactivityGraph.types) {
            return;
        }
        for (const type of this._interactivityGraph.types) {
            this._types.push(gltfTypeToBabylonType[type.signature]);
        }
    }
    _parseDeclarations() {
        if (!this._interactivityGraph.declarations) {
            return;
        }
        for (const declaration of this._interactivityGraph.declarations) {
            // make sure we have the mapping for this operation
            const mapping = getMappingForDeclaration(declaration);
            // mapping is defined, because we generate an empty mapping if it's not found
            if (!mapping) {
                Logger.Error(["No mapping found for declaration", declaration]);
                throw new Error("Error parsing declarations");
            }
            this._mappings.push({
                flowGraphMapping: mapping,
                fullOperationName: declaration.extension ? declaration.op + ":" + declaration.extension : declaration.op,
            });
        }
    }
    _parseVariables() {
        if (!this._interactivityGraph.variables) {
            return;
        }
        for (const variable of this._interactivityGraph.variables) {
            const parsed = this._parseVariable(variable);
            // set the default values here
            this._staticVariables.push(parsed);
        }
    }
    _parseVariable(variable, dataTransform) {
        const type = this._types[variable.type];
        if (!type) {
            Logger.Error(["No type found for variable", variable]);
            throw new Error("Error parsing variables");
        }
        if (variable.value) {
            if (variable.value.length !== type.length) {
                Logger.Error(["Invalid value length for variable", variable, type]);
                throw new Error("Error parsing variables");
            }
        }
        const value = variable.value || [];
        if (!value.length) {
            switch (type.flowGraphType) {
                case "boolean" /* FlowGraphTypes.Boolean */:
                    value.push(false);
                    break;
                case "FlowGraphInteger" /* FlowGraphTypes.Integer */:
                    value.push(0);
                    break;
                case "number" /* FlowGraphTypes.Number */:
                    value.push(NaN);
                    break;
                case "Vector2" /* FlowGraphTypes.Vector2 */:
                    value.push(NaN, NaN);
                    break;
                case "Vector3" /* FlowGraphTypes.Vector3 */:
                    value.push(NaN, NaN, NaN);
                    break;
                case "Vector4" /* FlowGraphTypes.Vector4 */:
                case "Matrix2D" /* FlowGraphTypes.Matrix2D */:
                case "Quaternion" /* FlowGraphTypes.Quaternion */:
                    value.fill(NaN, 0, 4);
                    break;
                case "Matrix" /* FlowGraphTypes.Matrix */:
                    value.fill(NaN, 0, 16);
                    break;
                case "Matrix3D" /* FlowGraphTypes.Matrix3D */:
                    value.fill(NaN, 0, 9);
                    break;
            }
        }
        // in case of NaN, Infinity, we need to parse the string to the object itself
        if (type.elementType === "number" && typeof value[0] === "string") {
            value[0] = parseFloat(value[0]);
        }
        return { type: type.flowGraphType, value: dataTransform ? dataTransform(value, this) : value };
    }
    _parseEvents() {
        if (!this._interactivityGraph.events) {
            return;
        }
        for (const event of this._interactivityGraph.events) {
            const converted = {
                eventId: event.id || "internalEvent_" + this._internalEventsCounter++,
            };
            if (event.values) {
                converted.eventData = Object.keys(event.values).map((key) => {
                    const eventValue = event.values?.[key];
                    if (!eventValue) {
                        Logger.Error(["No value found for event key", key]);
                        throw new Error("Error parsing events");
                    }
                    const type = this._types[eventValue.type];
                    if (!type) {
                        Logger.Error(["No type found for event value", eventValue]);
                        throw new Error("Error parsing events");
                    }
                    const value = typeof eventValue.value !== "undefined" ? this._parseVariable(eventValue) : undefined;
                    return {
                        id: key,
                        type: type.flowGraphType,
                        eventData: true,
                        value,
                    };
                });
            }
            this._events.push(converted);
        }
    }
    _parseNodes() {
        if (!this._interactivityGraph.nodes) {
            return;
        }
        for (const node of this._interactivityGraph.nodes) {
            // some validation
            if (typeof node.declaration !== "number") {
                Logger.Error(["No declaration found for node", node]);
                throw new Error("Error parsing nodes");
            }
            const mapping = this._mappings[node.declaration];
            if (!mapping) {
                Logger.Error(["No mapping found for node", node]);
                throw new Error("Error parsing nodes");
            }
            if (mapping.flowGraphMapping.validation) {
                const validationResult = mapping.flowGraphMapping.validation(node, this._interactivityGraph, this._gltf);
                if (!validationResult.valid) {
                    throw new Error(`Error validating interactivity node ${this._interactivityGraph.declarations?.[node.declaration].op} - ${validationResult.error}`);
                }
            }
            const blocks = [];
            // create block(s) for this node using the mapping
            for (const blockType of mapping.flowGraphMapping.blocks) {
                const block = this._getEmptyBlock(blockType, mapping.fullOperationName);
                this._parseNodeConfiguration(node, block, mapping.flowGraphMapping, blockType);
                blocks.push(block);
            }
            this._nodes.push({ blocks, fullOperationName: mapping.fullOperationName });
        }
    }
    _getEmptyBlock(className, type) {
        return {
            uniqueId: RandomGUID(),
            className,
            dataInputs: [],
            dataOutputs: [],
            signalInputs: [],
            signalOutputs: [],
            config: {},
            type,
            metadata: {},
        };
    }
    _parseNodeConfiguration(node, block, nodeMapping, blockType) {
        const gltfConfiguration = node.configuration;
        if (gltfConfiguration) {
            for (const key in gltfConfiguration) {
                const gltfProperty = gltfConfiguration[key];
                if (!gltfProperty) {
                    throw new Error("Error parsing node configuration");
                }
                const propertyMapping = nodeMapping.configuration?.[key];
                const belongsToBlock = propertyMapping && propertyMapping.toBlock ? propertyMapping.toBlock === blockType : nodeMapping.blocks.indexOf(blockType) === 0;
                if (belongsToBlock) {
                    let value = propertyMapping?.defaultValue;
                    if (gltfProperty?.value) {
                        value = gltfProperty.value;
                    }
                    if (!propertyMapping?.isArray) {
                        if (value.length !== 1) {
                            Logger.Warn(`Invalid non-array value length: ${value.length}`);
                        }
                        value = value[0];
                    }
                    if (propertyMapping?.dataTransformer) {
                        value = propertyMapping.dataTransformer(value, this);
                    }
                    if (value !== undefined) {
                        // Update the flow graph block config.
                        block.config[propertyMapping?.name || key] = {
                            value: value,
                        };
                    }
                }
            }
        }
    }
    _parseNodeConnections(context) {
        for (let i = 0; i < this._nodes.length; i++) {
            // get the corresponding gltf node
            const gltfNode = this._interactivityGraph.nodes?.[i];
            if (!gltfNode) {
                // should never happen but let's still check
                Logger.Error(["No node found for interactivity node", this._nodes[i]]);
                throw new Error("Error parsing node connections");
            }
            const flowGraphBlocks = this._nodes[i];
            const outputMapper = this._mappings[gltfNode.declaration];
            // validate
            if (!outputMapper) {
                Logger.Error(["No mapping found for node", gltfNode]);
                throw new Error("Error parsing node connections");
            }
            const flowsFromGLTF = gltfNode.flows || {};
            const flowsKeys = Object.keys(flowsFromGLTF).sort(); // sorting as some operations require sorted keys
            // connect the flows
            for (const flowKey of flowsKeys) {
                const flow = flowsFromGLTF[flowKey];
                const flowMapping = outputMapper.flowGraphMapping.outputs?.flows?.[flowKey];
                const socketOutName = flowMapping?.name || flowKey;
                // create a serialized socket
                const socketOut = this._createNewSocketConnection(socketOutName, true);
                const block = (flowMapping && flowMapping.toBlock && flowGraphBlocks.blocks.find((b) => b.className === flowMapping.toBlock)) || flowGraphBlocks.blocks[0];
                block.signalOutputs.push(socketOut);
                // get the input node of this block
                const inputNodeId = flow.node;
                const nodeIn = this._nodes[inputNodeId];
                if (!nodeIn) {
                    Logger.Error(["No node found for input node id", inputNodeId]);
                    throw new Error("Error parsing node connections");
                }
                // get the mapper for the input node - in case it mapped to multiple blocks
                const inputMapper = getMappingForFullOperationName(nodeIn.fullOperationName);
                if (!inputMapper) {
                    Logger.Error(["No mapping found for input node", nodeIn]);
                    throw new Error("Error parsing node connections");
                }
                let flowInMapping = inputMapper.inputs?.flows?.[flow.socket || "in"];
                let arrayMapping = false;
                if (!flowInMapping) {
                    for (const key in inputMapper.inputs?.flows) {
                        if (key.startsWith("[") && key.endsWith("]")) {
                            arrayMapping = true;
                            flowInMapping = inputMapper.inputs?.flows?.[key];
                        }
                    }
                }
                const nodeInSocketName = flowInMapping ? (arrayMapping ? flowInMapping.name.replace("$1", flow.socket || "") : flowInMapping.name) : flow.socket || "in";
                const inputBlock = (flowInMapping && flowInMapping.toBlock && nodeIn.blocks.find((b) => b.className === flowInMapping.toBlock)) || nodeIn.blocks[0];
                // in all of the flow graph input connections, find the one with the same name as the socket
                let socketIn = inputBlock.signalInputs.find((s) => s.name === nodeInSocketName);
                // if the socket doesn't exist, create the input socket for the connection
                if (!socketIn) {
                    socketIn = this._createNewSocketConnection(nodeInSocketName);
                    inputBlock.signalInputs.push(socketIn);
                }
                // connect the sockets
                socketIn.connectedPointIds.push(socketOut.uniqueId);
                socketOut.connectedPointIds.push(socketIn.uniqueId);
            }
            // connect the values
            const valuesFromGLTF = gltfNode.values || {};
            const valuesKeys = Object.keys(valuesFromGLTF);
            for (const valueKey of valuesKeys) {
                const value = valuesFromGLTF[valueKey];
                let valueMapping = outputMapper.flowGraphMapping.inputs?.values?.[valueKey];
                let arrayMapping = false;
                if (!valueMapping) {
                    for (const key in outputMapper.flowGraphMapping.inputs?.values) {
                        if (key.startsWith("[") && key.endsWith("]")) {
                            arrayMapping = true;
                            valueMapping = outputMapper.flowGraphMapping.inputs?.values?.[key];
                        }
                    }
                }
                const socketInName = valueMapping ? (arrayMapping ? valueMapping.name.replace("$1", valueKey) : valueMapping.name) : valueKey;
                // create a serialized socket
                const socketIn = this._createNewSocketConnection(socketInName);
                const block = (valueMapping && valueMapping.toBlock && flowGraphBlocks.blocks.find((b) => b.className === valueMapping.toBlock)) || flowGraphBlocks.blocks[0];
                block.dataInputs.push(socketIn);
                if (value.value !== undefined) {
                    const convertedValue = this._parseVariable(value, valueMapping && valueMapping.dataTransformer);
                    context._connectionValues[socketIn.uniqueId] = convertedValue;
                }
                else if (typeof value.node !== "undefined") {
                    const nodeOutId = value.node;
                    const nodeOutSocketName = value.socket || "value";
                    const nodeOut = this._nodes[nodeOutId];
                    if (!nodeOut) {
                        Logger.Error(["No node found for output socket reference", value]);
                        throw new Error("Error parsing node connections");
                    }
                    const outputMapper = getMappingForFullOperationName(nodeOut.fullOperationName);
                    if (!outputMapper) {
                        Logger.Error(["No mapping found for output socket reference", value]);
                        throw new Error("Error parsing node connections");
                    }
                    let valueMapping = outputMapper.outputs?.values?.[nodeOutSocketName];
                    let arrayMapping = false;
                    // check if there is an array mapping defined
                    if (!valueMapping) {
                        // search for a value mapping that has an array mapping
                        for (const key in outputMapper.outputs?.values) {
                            if (key.startsWith("[") && key.endsWith("]")) {
                                arrayMapping = true;
                                valueMapping = outputMapper.outputs?.values?.[key];
                            }
                        }
                    }
                    const socketOutName = valueMapping ? (arrayMapping ? valueMapping.name.replace("$1", nodeOutSocketName) : valueMapping?.name) : nodeOutSocketName;
                    const outBlock = (valueMapping && valueMapping.toBlock && nodeOut.blocks.find((b) => b.className === valueMapping.toBlock)) || nodeOut.blocks[0];
                    let socketOut = outBlock.dataOutputs.find((s) => s.name === socketOutName);
                    // if the socket doesn't exist, create it
                    if (!socketOut) {
                        socketOut = this._createNewSocketConnection(socketOutName, true);
                        outBlock.dataOutputs.push(socketOut);
                    }
                    // connect the sockets
                    socketIn.connectedPointIds.push(socketOut.uniqueId);
                    socketOut.connectedPointIds.push(socketIn.uniqueId);
                }
                else {
                    Logger.Error(["Invalid value for value connection", value]);
                    throw new Error("Error parsing node connections");
                }
            }
            // inter block connections
            if (outputMapper.flowGraphMapping.interBlockConnectors) {
                for (const connector of outputMapper.flowGraphMapping.interBlockConnectors) {
                    const input = connector.input;
                    const output = connector.output;
                    const isVariable = connector.isVariable;
                    this._connectFlowGraphNodes(input, output, flowGraphBlocks.blocks[connector.inputBlockIndex], flowGraphBlocks.blocks[connector.outputBlockIndex], isVariable);
                }
            }
            if (outputMapper.flowGraphMapping.extraProcessor) {
                const declaration = this._interactivityGraph.declarations?.[gltfNode.declaration];
                if (!declaration) {
                    Logger.Error(["No declaration found for extra processor", gltfNode]);
                    throw new Error("Error parsing node connections");
                }
                flowGraphBlocks.blocks = outputMapper.flowGraphMapping.extraProcessor(gltfNode, declaration, outputMapper.flowGraphMapping, this, flowGraphBlocks.blocks, context, this._gltf);
            }
        }
    }
    _createNewSocketConnection(name, isOutput) {
        return {
            uniqueId: RandomGUID(),
            name,
            _connectionType: isOutput ? 1 /* FlowGraphConnectionType.Output */ : 0 /* FlowGraphConnectionType.Input */,
            connectedPointIds: [],
        };
    }
    _connectFlowGraphNodes(input, output, serializedInput, serializedOutput, isVariable) {
        const inputArray = isVariable ? serializedInput.dataInputs : serializedInput.signalInputs;
        const outputArray = isVariable ? serializedOutput.dataOutputs : serializedOutput.signalOutputs;
        const inputConnection = inputArray.find((s) => s.name === input) || this._createNewSocketConnection(input);
        const outputConnection = outputArray.find((s) => s.name === output) || this._createNewSocketConnection(output, true);
        // of not found add it to the array
        if (!inputArray.find((s) => s.name === input)) {
            inputArray.push(inputConnection);
        }
        if (!outputArray.find((s) => s.name === output)) {
            outputArray.push(outputConnection);
        }
        // connect the sockets
        inputConnection.connectedPointIds.push(outputConnection.uniqueId);
        outputConnection.connectedPointIds.push(inputConnection.uniqueId);
    }
    getVariableName(index) {
        return "staticVariable_" + index;
    }
    serializeToFlowGraph() {
        const context = {
            uniqueId: RandomGUID(),
            _userVariables: {},
            _connectionValues: {},
        };
        this._parseNodeConnections(context);
        for (let i = 0; i < this._staticVariables.length; i++) {
            const variable = this._staticVariables[i];
            context._userVariables[this.getVariableName(i)] = variable;
        }
        const allBlocks = this._nodes.reduce((acc, val) => acc.concat(val.blocks), []);
        return {
            rightHanded: true,
            allBlocks,
            executionContexts: [context],
        };
    }
}

const NAME = "KHR_interactivity";
/**
 * Loader extension for KHR_interactivity
 */
class KHR_interactivity {
    /**
     * @internal
     * @param _loader
     */
    constructor(_loader) {
        this._loader = _loader;
        /**
         * The name of this extension.
         */
        this.name = NAME;
        this.enabled = this._loader.isExtensionUsed(NAME);
        this._pathConverter = GetPathToObjectConverter(this._loader.gltf);
        // avoid starting animations automatically.
        _loader._skipStartAnimationStep = true;
        // Update object model with new pointers
        const scene = _loader.babylonScene;
        if (scene) {
            _AddInteractivityObjectModel(scene);
        }
    }
    dispose() {
        this._loader = null;
        delete this._pathConverter;
    }
    // eslint-disable-next-line no-restricted-syntax, @typescript-eslint/no-misused-promises
    async onReady() {
        if (!this._loader.babylonScene || !this._pathConverter) {
            return;
        }
        const scene = this._loader.babylonScene;
        const interactivityDefinition = this._loader.gltf.extensions?.KHR_interactivity;
        if (!interactivityDefinition) {
            // This can technically throw, but it's not a critical error
            return;
        }
        const coordinator = new FlowGraphCoordinator({ scene });
        coordinator.dispatchEventsSynchronously = false; // glTF interactivity dispatches events asynchronously
        const graphs = interactivityDefinition.graphs.map((graph) => {
            const parser = new InteractivityGraphToFlowGraphParser(graph, this._loader.gltf, this._loader.parent.targetFps);
            return parser.serializeToFlowGraph();
        });
        // parse each graph async
        await Promise.all(graphs.map(async (graph) => await ParseFlowGraphAsync(graph, { coordinator, pathConverter: this._pathConverter })));
        coordinator.start();
    }
}
/**
 * @internal
 * populates the object model with the interactivity extension
 */
function _AddInteractivityObjectModel(scene) {
    // Note - all of those are read-only, as per the specs!
    // active camera rotation
    AddObjectAccessorToKey("/extensions/KHR_interactivity/?/activeCamera/rotation", {
        get: () => {
            if (!scene.activeCamera) {
                return new Quaternion(NaN, NaN, NaN, NaN);
            }
            const quat = Quaternion.FromRotationMatrix(scene.activeCamera.getWorldMatrix()).normalize();
            if (!scene.useRightHandedSystem) {
                quat.w *= -1; // glTF uses right-handed system, while babylon uses left-handed
                quat.x *= -1; // glTF uses right-handed system, while babylon uses left-handed
            }
            return quat;
        },
        type: "Quaternion",
        getTarget: () => scene.activeCamera,
    });
    // activeCamera position
    AddObjectAccessorToKey("/extensions/KHR_interactivity/?/activeCamera/position", {
        get: () => {
            if (!scene.activeCamera) {
                return new Vector3(NaN, NaN, NaN);
            }
            const pos = scene.activeCamera.getWorldMatrix().getTranslation(); // not global position
            if (!scene.useRightHandedSystem) {
                pos.x *= -1; // glTF uses right-handed system, while babylon uses left-handed
            }
            return pos;
        },
        type: "Vector3",
        getTarget: () => scene.activeCamera,
    });
    // /animations/{} pointers:
    AddObjectAccessorToKey("/animations/{}/extensions/KHR_interactivity/isPlaying", {
        get: (animation) => {
            return animation._babylonAnimationGroup?.isPlaying ?? false;
        },
        type: "boolean",
        getTarget: (animation) => {
            return animation._babylonAnimationGroup;
        },
    });
    AddObjectAccessorToKey("/animations/{}/extensions/KHR_interactivity/minTime", {
        get: (animation) => {
            return (animation._babylonAnimationGroup?.from ?? 0) / 60; // fixed factor for duration-to-frames conversion
        },
        type: "number",
        getTarget: (animation) => {
            return animation._babylonAnimationGroup;
        },
    });
    AddObjectAccessorToKey("/animations/{}/extensions/KHR_interactivity/maxTime", {
        get: (animation) => {
            return (animation._babylonAnimationGroup?.to ?? 0) / 60; // fixed factor for duration-to-frames conversion
        },
        type: "number",
        getTarget: (animation) => {
            return animation._babylonAnimationGroup;
        },
    });
    // playhead
    AddObjectAccessorToKey("/animations/{}/extensions/KHR_interactivity/playhead", {
        get: (animation) => {
            return (animation._babylonAnimationGroup?.getCurrentFrame() ?? 0) / 60; // fixed factor for duration-to-frames conversion
        },
        type: "number",
        getTarget: (animation) => {
            return animation._babylonAnimationGroup;
        },
    });
    //virtualPlayhead - TODO, do we support this property in our animations? getCurrentFrame  is the only method we have for this.
    AddObjectAccessorToKey("/animations/{}/extensions/KHR_interactivity/virtualPlayhead", {
        get: (animation) => {
            return (animation._babylonAnimationGroup?.getCurrentFrame() ?? 0) / 60; // fixed factor for duration-to-frames conversion
        },
        type: "number",
        getTarget: (animation) => {
            return animation._babylonAnimationGroup;
        },
    });
}
// Register flow graph blocks. Do it here so they are available when the extension is enabled.
addToBlockFactory(NAME, "FlowGraphGLTFDataProvider", async () => {
    return (await __vitePreload(async () => { const {FlowGraphGLTFDataProvider} = await import('./flowGraphGLTFDataProvider-BF36Cs2j.js');return { FlowGraphGLTFDataProvider }},true              ?__vite__mapDeps([58,1,2,3,4,7]):void 0,import.meta.url)).FlowGraphGLTFDataProvider;
});
unregisterGLTFExtension(NAME);
registerGLTFExtension(NAME, true, (loader) => new KHR_interactivity(loader));

const KHR_interactivity$1 = /*#__PURE__*/Object.freeze(/*#__PURE__*/Object.defineProperty({
    __proto__: null,
    KHR_interactivity,
    _AddInteractivityObjectModel
}, Symbol.toStringTag, { value: 'Module' }));

export { FlowGraphBlock as F, GetFlowGraphAssetWithType as G, KHR_interactivity$1 as K, _IsDescendantOf as _, FlowGraphAsyncExecutionBlock as a, FlowGraphExecutionBlockWithOutSignal as b, FlowGraphEventBlock as c, FlowGraphCoordinator as d, FlowGraphExecutionBlock as e, defaultValueSerializationFunction as f, getNumericValue as g, _GetClassNameOf as h, isNumeric as i, _AreSameVectorOrQuaternionClass as j, _AreSameMatrixClass as k, _AreSameIntegerClass as l };
//# sourceMappingURL=KHR_interactivity-BPhVhH5L.js.map
